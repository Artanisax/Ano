from pathlib import Path
from copy import deepcopy
from pprint import pprint
import torch

print(torch.__version__)

def to_python(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return {'shape': tuple(v.shape), 'dtype': str(v.dtype)}
    return v

def summarize_callbacks(callbacks_dict):
    rows = []
    for key, state in callbacks_dict.items():
        rows.append({
            'state_key': key,
            'fields': sorted(state.keys()),
            'state': {k: to_python(v) for k, v in state.items()}
        })
    return rows

def matches_target(state_key, substrings):
    return any(s in state_key for s in substrings)

def reset_best_score_like(old_value, mode):
    reset_value = float('inf') if mode == 'min' else float('-inf')
    if isinstance(old_value, torch.Tensor):
        return old_value.new_tensor(reset_value)
    return reset_value

SRC_CKPT = Path('runs/Ano/version_1/checkpoints/last.ckpt')
DST_CKPT = Path('runs/Ano/version_1/checkpoints/last_for_resume.ckpt')

PATCH_MODE = 'reset_early_stopping'  # 'reset_early_stopping' | 'drop_selected_callbacks' | 'drop_all_callbacks'

# 用于匹配 callback key，例如 EarlyStopping、ModelCheckpoint
TARGET_SUBSTRINGS = ['EarlyStopping']

# reset_early_stopping 模式下的开关
RESET_WAIT_COUNT = False
RESET_STOPPED_EPOCH = True
RESET_BEST_SCORE = False   # 如果你想完全重新开始 early stopping 计数，可改成 True
NEW_PATIENCE = 25          # 不想改就设为 None
OVERRIDE_MODE = None       # None / 'min' / 'max'
OVERRIDE_MONITOR = None    # 例如 'val/rec'

assert SRC_CKPT.exists(), f'源 checkpoint 不存在: {SRC_CKPT}'
DST_CKPT.parent.mkdir(parents=True, exist_ok=True)

print('SRC_CKPT =', SRC_CKPT)
print('DST_CKPT =', DST_CKPT)

ckpt = torch.load(SRC_CKPT)
print('Top-level keys:')
pprint(sorted(ckpt.keys()))

print('\nglobal_step =', ckpt.get('global_step'))
print('epoch       =', ckpt.get('epoch'))
print('has callbacks =', 'callbacks' in ckpt)
print('num callback states =', len(ckpt.get('callbacks', {})))

callback_rows = summarize_callbacks(ckpt.get('callbacks', {}))
for i, row in enumerate(callback_rows):
    print(f'[{i}] {row["state_key"]}')
    pprint(row['state'])
    print('-' * 100)

print('PATCH_MODE =', PATCH_MODE)
print('TARGET_SUBSTRINGS =', TARGET_SUBSTRINGS)

edited = deepcopy(ckpt)
callbacks = deepcopy(edited.get('callbacks', {}))

before = deepcopy(callbacks)

if PATCH_MODE == 'drop_all_callbacks':
    callbacks = {}

elif PATCH_MODE == 'drop_selected_callbacks':
    callbacks = {
        key: state
        for key, state in callbacks.items()
        if not matches_target(key, TARGET_SUBSTRINGS)
    }

elif PATCH_MODE == 'reset_early_stopping':
    for key, state in callbacks.items():
        if not matches_target(key, TARGET_SUBSTRINGS):
            continue

        if RESET_WAIT_COUNT and 'wait_count' in state:
            state['wait_count'] = 0

        if RESET_STOPPED_EPOCH and 'stopped_epoch' in state:
            state['stopped_epoch'] = 0

        if NEW_PATIENCE is not None and 'patience' in state:
            state['patience'] = int(NEW_PATIENCE)

        if OVERRIDE_MODE is not None and 'mode' in state:
            state['mode'] = OVERRIDE_MODE

        if OVERRIDE_MONITOR is not None and 'monitor' in state:
            state['monitor'] = OVERRIDE_MONITOR

        if RESET_BEST_SCORE and 'best_score' in state:
            mode = state.get('mode', 'min') if OVERRIDE_MODE is None else OVERRIDE_MODE
            state['best_score'] = reset_best_score_like(state['best_score'], mode)
else:
    raise ValueError(f'Unknown PATCH_MODE: {PATCH_MODE}')

edited['callbacks'] = callbacks
print('修改完成。')

print('===== BEFORE =====')
for key, state in before.items():
    if matches_target(key, TARGET_SUBSTRINGS):
        print(key)
        pprint({k: to_python(v) for k, v in state.items()})
        print('-' * 100)

print('\n===== AFTER =====')
for key, state in edited.get('callbacks', {}).items():
    if matches_target(key, TARGET_SUBSTRINGS):
        print(key)
        pprint({k: to_python(v) for k, v in state.items()})
        print('-' * 100)

torch.save(edited, DST_CKPT)
print(f'已保存到: {DST_CKPT}')
