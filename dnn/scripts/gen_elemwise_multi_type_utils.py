# As cuda currently do not support quint8, so we just ignore it.
SUPPORT_DTYPES = [('dt_qint8', 'dt_qint8')]
SUPPORT_QINT32_DTYPES = [('dt_qint32', 'dt_qint8'), ('dt_qint8', 'dt_qint32'),
                         ('dt_qint4', 'dt_qint32'), ('dt_quint4', 'dt_qint32')]

SUPPORT_DTYPES_Q4 = [('dt_qint4', 'dt_qint4'), ('dt_quint4', 'dt_quint4')]
SUPPORT_QINT32_DTYPES_Q4 = [('dt_qint32', 'dt_qint4'), ('dt_qint32', 'dt_quint4')]

MODES = {
    1: ['RELU', 'ABS', 'NEGATE', 'ACOS', 'ASIN', 'CEIL', 'COS',
        'EXP', 'EXPM1', 'FLOOR', 'LOG', 'LOG1P', 'SIGMOID', 'SIN',
        'TANH', 'FAST_TANH', 'ROUND', 'ERF', 'ERFINV', 'ERFC',
        'ERFCINV', 'H_SWISH', 'SILU', 'GELU'],
    2: ['ABS_GRAD', 'ADD', 'FLOOR_DIV', 'MAX', 'MIN', 'MOD', 'MUL',
        'SIGMOID_GRAD', 'SUB', 'SWITCH_GT0', 'TANH_GRAD', 'LT',
        'LEQ', 'EQ', 'FUSE_ADD_RELU', 'TRUE_DIV', 'POW',
        'LOG_SUM_EXP', 'FUSE_ADD_TANH', 'FAST_TANH_GRAD',
        'FUSE_ADD_SIGMOID', 'ATAN2', 'H_SWISH_GRAD',
        'FUSE_ADD_H_SWISH', 'SILU_GRAD', 'GELU_GRAD'],
    3: ['COND_LEQ_MOV', 'FUSE_MUL_ADD3'],
}

QINT4_MODES = {
    1: ['RELU', 'ABS', 'NEGATE', 'CEIL', 'FLOOR', 'SIGMOID',
        'TANH', 'FAST_TANH', 'ROUND', 'H_SWISH'],
    2: ['ADD', 'MAX', 'MIN', 'MUL', 'SUB', 'SWITCH_GT0', 
        'LT', 'LEQ', 'EQ', 'FUSE_ADD_RELU', 'FUSE_ADD_TANH', 
        'FUSE_ADD_SIGMOID', 'FUSE_ADD_H_SWISH'],
    3: ['COND_LEQ_MOV', 'FUSE_MUL_ADD3'],
}

QINT32_MODES = {
    1: ['RELU', 'SIGMOID', 'TANH', 'FAST_TANH', 'H_SWISH'],
    2: ['ADD', 'FUSE_ADD_RELU', 'FUSE_ADD_SIGMOID',
        'FUSE_ADD_TANH', 'FUSE_ADD_H_SWISH']
}
