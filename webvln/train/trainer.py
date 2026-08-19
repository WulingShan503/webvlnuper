"""训练与验证循环（论文 3.9 / 5.1 节）。

官方 ``agent.py`` 的一次迭代（``--feedback mix``）：

    zero_grad
    rollout(feedback='sample')   两趟的损失都累进同一个 loss
    rollout(feedback='teacher')
    loss.backward()
    clip_grad_norm(40.)
    step

验证调度按论文 3.9 节：140,000 迭代后每 1,000 步验证一次，
按 Best Score = SR + WUPS0.9 保存最优权重。

需要 torch，故本机未做运行时验证；下标对齐与停止判定等易错逻辑
已在 ``env.py`` / ``rollout.py`` 层测过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from webvln.data.dataset import WebVLNDataset
from webvln.eval.metrics import Result, score_results
from webvln.models.config import WebVLNConfig
from webvln.train.batching import (
    build_answer_tensors,
    build_candidate_tensor,
    build_language_tensors,
    build_teacher_tensor,
    candidate_lengths,
)
from webvln.train.env import Observation, WebVLNEnv
from webvln.train.rollout import (
    FEEDBACK_ARGMAX,
    FEEDBACK_MIX,
    FEEDBACK_SAMPLE,
    FEEDBACK_TEACHER,
    MIX_PASSES,
    RolloutRecorder,
    resolve_action,
)


@dataclass
class RolloutOutput:
    """一趟 rollout 的产物。

    Attributes:
        recorder: 轨迹与结束状态。
        nav_logits: 各步的动作 logits。
        nav_targets: 各步的教师动作。
        answer_states: 各样本停止那一步的状态，作为回答头的条件。
    """

    recorder: RolloutRecorder
    nav_logits: List[Any] = field(default_factory=list)
    nav_targets: List[Any] = field(default_factory=list)
    answer_states: Optional[Any] = None


class Trainer:
    """WebVLN-Net 的训练器。

    Attributes:
        model: WebVLNNet。
        env: 训练环境。
        config: 超参。
    """

    def __init__(
        self,
        model: Any,
        env: WebVLNEnv,
        config: Optional[WebVLNConfig] = None,
        loss_fn: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.env = env
        self.config = config or WebVLNConfig()

        if loss_fn is None:
            from webvln.models.losses import WebVLNLoss

            loss_fn = WebVLNLoss(self.config)
        self.loss_fn = loss_fn
        self.optimizer = None
        self.best_score = -1.0
        self.history: List[Dict[str, Any]] = []

    def build_optimizer(self) -> Any:
        """构建 AdamW。

        官方对 ``vln_bert`` 与 ``qa_decoder`` 各建一个优化器并分别 step；
        这里合成一个参数组——两者用同一学习率与 weight decay，
        分开只是官方的组织方式，不影响梯度。
        """
        import torch

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return self.optimizer

    # --- rollout ------------------------------------------------------------

    def rollout(
        self, batch: Sequence[Any], feedback: str = FEEDBACK_TEACHER
    ) -> RolloutOutput:
        """跑一批 episode。

        Args:
            batch: Episode 列表。
            feedback: teacher / argmax / sample 之一。``mix`` 请在
                ``train_step`` 中展开，单趟 rollout 不接受它。

        Returns:
            RolloutOutput。
        """
        import torch

        if feedback == FEEDBACK_MIX:
            raise ValueError("mix 需展开为两趟 rollout，见 train_step")

        obs = self.env.reset(batch)
        recorder = RolloutRecorder(obs)

        input_ids, lang_mask = build_language_tensors(obs, self.config.pad_token_id)
        # 语言只在初始化时编码一次（论文 3.2 节）：初始化得到的 token
        # 已是 Q&D 的良好表示，导航中重编码只增开销。
        lang_feats, state = self.model.encode_language(input_ids, lang_mask)

        nav_logits: List[Any] = []
        nav_targets: List[Any] = []
        # 各步的状态都留着：回答阶段要取「停止那一步」的状态，
        # 而各样本停止的步号不同。
        states_per_step: List[Any] = []

        for step in range(self.config.max_action_len):
            cand_feats = build_candidate_tensor(
                obs, self.config.feature_size, self.config.tokens_per_candidate
            )
            lengths = candidate_lengths(obs)

            state, logits, _ = self.model.navigate_step(
                lang_feats=lang_feats,
                lang_attention_mask=lang_mask,
                candidate_feats=cand_feats,
                action_lengths=lengths,
                state=state if step > 0 else None,
            )
            states_per_step.append(state)

            target = build_teacher_tensor(obs, recorder.ended, self.config.ignore_id)
            nav_logits.append(logits)
            nav_targets.append(target)

            actions = self._select_actions(logits, target, feedback)
            decisions = [
                resolve_action(int(actions[i]), obs[i], recorder.ended[i],
                               self.config.ignore_id)
                for i in range(len(obs))
            ]
            env_actions = recorder.record(step, decisions, obs)

            if recorder.all_ended():
                break
            obs = self.env.step(env_actions)

        recorder.finalize(max_step=len(states_per_step) - 1)
        # 逐样本取「停止那一步」的状态：各样本停止步号不同，
        # 统一取最后一步会让早停的样本带着之后若干步的状态去作答。
        answer_states = torch.stack(
            [
                states_per_step[recorder.answer_step[i]][i]
                for i in range(len(recorder.idxs))
            ]
        )

        return RolloutOutput(
            recorder=recorder,
            nav_logits=nav_logits,
            nav_targets=nav_targets,
            answer_states=answer_states,
        )

    def _select_actions(self, logits: Any, target: Any, feedback: str) -> Any:
        """按 feedback 策略选动作。"""
        import torch
        import torch.nn.functional as F

        if feedback == FEEDBACK_TEACHER:
            return target
        if feedback == FEEDBACK_ARGMAX:
            return logits.max(1)[1].detach()
        if feedback == FEEDBACK_SAMPLE:
            probs = F.softmax(logits, 1)
            return torch.distributions.Categorical(probs).sample().detach()
        raise ValueError(f"未知的 feedback：{feedback!r}")

    # --- 优化 ---------------------------------------------------------------

    def train_step(self, batch: Sequence[Any], feedback: str = FEEDBACK_MIX) -> Dict[str, float]:
        """一次迭代：前向、反传、更新。

        ``mix`` 展开为 sample 与 teacher 两趟，两趟损失累加后一次反传——
        这是官方的做法，不是按概率二选一。

        Returns:
            含 total / nav / ans 的损失字典（已 detach）。
        """
        import torch

        if self.optimizer is None:
            self.build_optimizer()

        self.model.train()
        self.optimizer.zero_grad()

        passes = MIX_PASSES if feedback == FEEDBACK_MIX else ((feedback, 1.0),)
        total = None
        parts = {"nav": 0.0, "ans": 0.0}

        for name, weight in passes:
            out = self.rollout(batch, feedback=name)
            # 答案序列取自 episode 本身，与智能体最终停在哪一页无关。
            # 用 rollout 结束后的观测去取会拿到「当前页面」的记录，
            # 那时 batch 顺序虽未变，语义上却是错的。
            answer_input, answer_targets = build_answer_tensors(out.recorder.obs)
            answer_logits = self.model.answer(out.answer_states, answer_input)

            losses = self.loss_fn(
                nav_logits_per_step=out.nav_logits,
                nav_targets_per_step=out.nav_targets,
                answer_logits=answer_logits,
                answer_targets=answer_targets,
                batch_size=len(batch),
            )
            scaled = losses["total"] * weight
            total = scaled if total is None else total + scaled
            parts["nav"] += float(losses["nav"].detach())
            parts["ans"] += float(losses["ans"].detach())

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {"total": float(total.detach()), **parts}

    # --- 验证 ---------------------------------------------------------------

    def validate(
        self, dataset: WebVLNDataset, wups_fn: Optional[Any] = None
    ) -> Dict[str, float]:
        """在一个划分上评测。

        用 argmax feedback 遍历整个划分一次（``iter_batches`` 不重复样本），
        再交给 ``eval.score_results`` 汇总。
        """
        import torch

        self.model.eval()
        results: List[Result] = []
        ground_truth = {ep.idx: ep for ep in dataset.episodes}

        with torch.no_grad():
            for batch in dataset.iter_batches():
                out = self.rollout(batch, feedback=FEEDBACK_ARGMAX)
                answers = self.model.generate_answer(
                    out.answer_states,
                    bos_token_id=self.config.answer_bos_id,
                    eos_token_id=self.config.answer_eos_id,
                )
                for i, record in enumerate(out.recorder.as_results()):
                    record["answer"] = self._decode(answers[i])
                    results.append(Result(**record))

        scores = score_results(results, ground_truth, wups_fn=wups_fn)
        return scores.as_dict()

    def _decode(self, token_ids: Any) -> str:
        """把生成的 token id 解码成文本。

        tokenizer 由外部通过 ``self.tokenizer`` 注入；未注入时返回空串，
        使导航指标（SR / SPL / TL）仍可评测而不被 WUPS 阻塞。
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return ""
        return tokenizer.decode(
            [int(t) for t in token_ids], skip_special_tokens=True
        ).strip()

    def should_validate(self, iteration: int) -> bool:
        """是否到了验证时机。

        论文 3.9 节：140,000 迭代之后每 1,000 步验证一次。
        前 140,000 步不验证——早期模型的分数没有选择价值，
        而每次验证要跑完 1,262 条验证集，开销可观。
        """
        if iteration < self.config.eval_start_iter:
            return False
        return iteration % self.config.eval_interval == 0

    def update_best(self, summary: Dict[str, float]) -> bool:
        """按 Best Score = SR + WUPS0.9 判断是否为新最优。

        Returns:
            True 表示应保存权重。
        """
        score = summary.get("SR", 0.0) + summary.get("WUPS0.9", 0.0)
        if score > self.best_score:
            self.best_score = score
            return True
        return False

    # --- 驱动循环 -----------------------------------------------------------

    def fit(
        self,
        train_set: WebVLNDataset,
        val_sets: Optional[Dict[str, WebVLNDataset]] = None,
        n_iters: Optional[int] = None,
        feedback: str = FEEDBACK_MIX,
        save_path: Optional[str] = None,
        log_every: int = 1_000,
    ) -> List[Dict[str, Any]]:
        """完整训练循环。

        Args:
            train_set: 训练集。
            val_sets: ``{名称: 数据集}``，官方为 ``{"val": ..., "test": ...}``。
            n_iters: 迭代数，默认 ``config.max_iters``（200,000）。
            feedback: 默认 ``mix``（官方 ``run/train.bash`` 的设置）。
            save_path: 最优权重的保存路径。
            log_every: 日志间隔。

        Returns:
            history，每项含 iteration、损失与各划分的指标。

        Note:
            按**迭代数**而非 epoch 计数：``next_minibatch`` 在数据末尾回绕，
            因此不存在「跑完一轮」的概念，与官方一致。
        """
        n_iters = n_iters or self.config.max_iters
        val_sets = val_sets or {}

        for iteration in range(1, n_iters + 1):
            batch = train_set.next_minibatch()
            losses = self.train_step(batch, feedback=feedback)

            entry: Dict[str, Any] = {"iteration": iteration, **losses}

            if self.should_validate(iteration):
                for name, dataset in val_sets.items():
                    entry[name] = self.validate(dataset)
                # 模型选择只看验证集：用测试集选模型等于把测试集当验证集，
                # 报告的 Test SR 就不再是对未见数据的估计。
                if "val" in entry and self.update_best(entry["val"]):
                    entry["best"] = True
                    if save_path:
                        self.save(save_path, iteration)

            if iteration % log_every == 0 or "best" in entry:
                self.history.append(entry)

        return self.history

    def save(self, path: str, iteration: int = 0) -> None:
        """保存权重与迭代数，对应官方 ``best_val`` 快照。"""
        import torch

        torch.save(
            {
                "iteration": iteration,
                "best_score": self.best_score,
                "model": self.model.state_dict(),
                "config": self.config.as_dict(),
            },
            path,
        )

    def load(self, path: str) -> int:
        """加载权重，返回其迭代数（官方 ``listner.load`` 的语义）。"""
        import torch

        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt["model"])
        self.best_score = ckpt.get("best_score", -1.0)
        return int(ckpt.get("iteration", 0))
