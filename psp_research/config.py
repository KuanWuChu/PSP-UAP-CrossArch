"""Strict research contract. Unknown fields and coercible strings are errors."""
import json
import math
from pathlib import Path

DEFAULTS = dict(protocol_id='research_fixed_budget_v1', surrogate='resnet152',
    surrogate_weights='ResNet152_Weights.IMAGENET1K_V1', epsilon=10/255, delta_size=224,
    init='uniform', optimizer='Adam', lr=0.1, max_steps=10000,
    checkpoint_selection='final_iterate', seeds=[126, 2025, 3407], prior='jigsaw',
    prior_behavior='upstream_unchanged', p_active=True, p_rate=0.65,
    num_copies_per_prior=10, aspect_ratio=[0.75, 4/3], input_transform=False,
    re_weight=False, source_mode='noise_crops', crop_area=[0.08, 1.0],
    normalization_space_for_delta='raw_RGB_0_1', target_tuning=False,
    variant='e2_noise_crops', temper=3.0, checkpoint_every=200,
    angle=6.0, scale_t_low=0.8, scale_t_high=4.0, shuffle_block=2)


def validate(config):
    if set(config) - set(DEFAULTS):
        raise ValueError('unknown config fields: ' + ','.join(sorted(set(config) - set(DEFAULTS))))
    if set(DEFAULTS) - set(config):
        raise ValueError('missing config fields: ' + ','.join(sorted(set(DEFAULTS) - set(config))))
    c = dict(config)
    for k in ['p_active', 'input_transform', 're_weight', 'target_tuning']:
        if type(c[k]) is not bool:
            raise ValueError(k + ' must be boolean')
    for k in ['delta_size', 'max_steps', 'num_copies_per_prior', 'checkpoint_every', 'shuffle_block']:
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(k + ' must be positive integer')
    for k in ['epsilon', 'lr', 'p_rate', 'temper', 'angle', 'scale_t_low', 'scale_t_high']:
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(k + ' must be finite positive number')
    if c['epsilon'] > 1 or c['p_rate'] > 1 or c['scale_t_low'] > c['scale_t_high']:
        raise ValueError('epsilon/p_rate/scaling out of range')
    for k in ['crop_area', 'aspect_ratio']:
        v = c[k]
        if (not isinstance(v, list) or len(v) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in v) or not 0 < v[0] <= v[1]):
            raise ValueError('invalid ' + k)
    if c['crop_area'][1] > 1 or c['seeds'] != [126, 2025, 3407] or any(type(x) is not int for x in c['seeds']):
        raise ValueError('crop area/seeds violate fixed protocol')
    for k, value in [('protocol_id', 'research_fixed_budget_v1'), ('surrogate', 'resnet152'),
                     ('surrogate_weights', 'ResNet152_Weights.IMAGENET1K_V1'), ('delta_size', 224),
                     ('init', 'uniform'), ('optimizer', 'Adam'), ('checkpoint_selection', 'final_iterate'),
                     ('prior_behavior', 'upstream_unchanged'), ('normalization_space_for_delta', 'raw_RGB_0_1'),
                     ('target_tuning', False), ('p_active', True)]:
        if c[k] != value:
            raise ValueError('unsupported ' + k)
    if c['prior'] not in {'gauss', 'jigsaw'}:
        raise ValueError('invalid prior')
    if c['source_mode'] not in {'noise_crops', 'psp_crops'}:
        raise ValueError('invalid source mode')
    variants = {'e2_noise_crops': ('noise_crops', False, [0.08, 1.0]),
                'e2_psp_only': ('psp_crops', False, [0.08, 1.0]),
                'full_original': ('psp_crops', True, [0.08, 1.0]),
                'full_local': ('psp_crops', True, [0.08, 0.35]),
                'full_large': ('psp_crops', True, [0.5, 1.0])}
    if c['variant'] not in variants:
        raise ValueError('invalid variant')
    source, full, area = variants[c['variant']]
    if (c['source_mode'], c['input_transform'], c['re_weight'], c['crop_area']) != (source, full, full, area):
        raise ValueError('variant/source/T/RW/crop contract mismatch')
    if c['aspect_ratio'] != [0.75, 4/3]:
        raise ValueError('aspect ratio fixed for all comparisons')
    return c


def load_config(path):
    return validate(json.loads(Path(path).read_text()))
