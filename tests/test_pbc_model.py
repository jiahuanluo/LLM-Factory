"""PBC model 单元测试：前向 + 反向。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from pbc_credit.collator import PbcCollator
from pbc_credit.model import PbcCreditModel, PbcCreditModelConfig
from pbc_credit.masking import add_masks_to_batch
from pbc_credit.losses import pretrain_loss, finetune_loss
from pbc_credit.fields import PAYSTATE_VOCAB_SIZE, PUBLIC_TYPE_VOCAB_SIZE


def _small_model_cfg(pretrain: bool = False) -> PbcCreditModelConfig:
    return PbcCreditModelConfig(
        d=32, n_heads=4, n_layers=1, dropout=0.0, top_hidden=64,
        use_text_branch=False,  # 单测只测结构分支，避免加载 gte-new
        user_numeric_dim=10,
        user_cat_tables={'性别代码表': 4, '学历代码表': 8, '学位代码表': 6,
                         '就业状况代码表': 15, '世界各国和地区名称代码': 7,
                         '婚姻状况代码表': 6, '单位性质代码表': 6,
                         '国民经济行业代码表': 21, '职业代码表': 9,
                         '职务代码表': 5, '居住状况代码表': 10},
        summary_numeric_dim=36,
        summary_cat_tables={'个人信贷交易提示业务类型代码表': 6,
                            '业务大类': 3,
                            '个人被追偿汇总信息业务类型代码表': 2,
                            '个人逾期（透支）汇总信息账户类型代码表': 5,
                            '相关还款责任人类型代码表': 6,
                            '个人 借贷交易相关还款责任类型代码表': 2,
                            '后付费业务类型代码表': 2,
                            '公共信息类型代码表': 4},
        account_numeric_dim=8,
        account_cat_tables={'个人借贷账户类型代码表': 6,
                            '个人借贷交易业务种类代码表': 24,
                            '个人借贷交易担保方式代码表': 8,
                            '币种代码表': 18,
                            '个人借贷交易还款频率代码表': 12},
        paystate_vocab_size=PAYSTATE_VOCAB_SIZE,
        query_numeric_dim=1,
        query_cat_tables={'机构类型代码': 18, '查询原因代码表': 15},
        public_type_vocab_size=PUBLIC_TYPE_VOCAB_SIZE,
    )


def _fake_sample(n_d1, n_r2, n_q, n_p):
    return {
        'user_numeric': torch.randn(10),
        'user_cat_ids': torch.randint(0, 4, (11,)),
        'user_cat_mask': torch.ones(11, dtype=torch.long),
        'summary_numeric': torch.randn(36),
        'summary_cat_ids': torch.randint(0, 3, (10,)),
        'summary_cat_mask': torch.ones(10, dtype=torch.long),
        'd1_numeric': torch.randn(n_d1, 8),
        'd1_cat_ids': torch.randint(0, 6, (n_d1, 6)),
        'd1_cat_mask': torch.ones(n_d1, 6, dtype=torch.long),
        'd1_paystate': torch.randint(0, PAYSTATE_VOCAB_SIZE, (n_d1, 60)),
        'd1_mask': torch.ones(n_d1, dtype=torch.long),
        'r1_numeric': torch.zeros(0, 8),
        'r1_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r1_cat_mask': torch.zeros(0, 6, dtype=torch.long),
        'r1_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r1_mask': torch.zeros(0, dtype=torch.long),
        'r2_numeric': torch.randn(n_r2, 8),
        'r2_cat_ids': torch.randint(0, 6, (n_r2, 6)),
        'r2_cat_mask': torch.ones(n_r2, 6, dtype=torch.long),
        'r2_paystate': torch.randint(0, PAYSTATE_VOCAB_SIZE, (n_r2, 60)),
        'r2_mask': torch.ones(n_r2, dtype=torch.long),
        'r3_numeric': torch.zeros(0, 8),
        'r3_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r3_cat_mask': torch.zeros(0, 6, dtype=torch.long),
        'r3_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r3_mask': torch.zeros(0, dtype=torch.long),
        'r4_numeric': torch.zeros(0, 8),
        'r4_cat_ids': torch.zeros(0, 6, dtype=torch.long),
        'r4_cat_mask': torch.zeros(0, 6, dtype=torch.long),
        'r4_paystate': torch.zeros(0, 60, dtype=torch.long),
        'r4_mask': torch.zeros(0, dtype=torch.long),
        'query_numeric': torch.randn(n_q, 1),
        'query_cat_ids': torch.randint(0, 5, (n_q, 2)),
        'query_cat_mask': torch.ones(n_q, 2, dtype=torch.long),
        'query_mask': torch.ones(n_q, dtype=torch.long),
        'public_numeric': torch.randn(n_p, 2),
        'public_cat_ids': torch.randint(0, 8, (n_p, 1)),
        'public_cat_mask': torch.ones(n_p, 1, dtype=torch.long),
        'public_mask': torch.ones(n_p, dtype=torch.long),
        'target': torch.tensor([float(n_d1 % 2)]),
    }


def test_finetune_forward_backward():
    """finetune 模式下前向输出 [B,1]；反向更新参数。"""
    torch.manual_seed(0)
    cfg = _small_model_cfg(pretrain=False)
    model = PbcCreditModel(cfg, pretrain_mode=False)

    samples = [_fake_sample(3, 2, 5, 2), _fake_sample(1, 4, 8, 0)]
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (2, 1), logits.shape

    # 反向
    loss = finetune_loss(logits, batch['target'], pos_weight=4.0)
    loss.backward()
    # 至少 user_encoder 的 norm weight 有 grad
    assert model.user_encoder.norm.weight.grad is not None


def test_pretrain_forward_backward():
    """pretrain 模式下 mask 损失可反传。"""
    torch.manual_seed(0)
    cfg = _small_model_cfg(pretrain=True)
    model = PbcCreditModel(cfg, pretrain_mode=True)

    samples = [_fake_sample(4, 2, 6, 1), _fake_sample(2, 3, 5, 0)]
    base_batch = PbcCollator()(samples)
    batch = add_masks_to_batch(base_batch, mask_ratio=0.5)
    out = model(batch)

    # 至少有一个分支的 pred
    assert any(k.endswith('_pred') for k in out), 'pretrain out should have pred keys'

    loss, comps = pretrain_loss(out)
    assert loss.item() > 0 or len(out) > 0  # loss 可能为 0 但不应 NaN
    loss.backward()
    assert not torch.isnan(loss)


def test_handles_empty_accounts():
    """所有账户分支为空时不崩。"""
    torch.manual_seed(0)
    cfg = _small_model_cfg(pretrain=False)
    model = PbcCreditModel(cfg, pretrain_mode=False)
    samples = [_fake_sample(0, 0, 2, 0), _fake_sample(0, 0, 1, 0)]
    batch = PbcCollator()(samples)
    logits = model(batch)
    assert logits.shape == (2, 1)
    assert not torch.isnan(logits).any()


if __name__ == '__main__':
    test_finetune_forward_backward()
    test_pretrain_forward_backward()
    test_handles_empty_accounts()
    print('✓ test_pbc_model passed')
