# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import amp
from megengine.core.tensor import amp as origin_amp


def test_grad_scaler():
    def check(enabled, low, high):
        assert amp.enabled == enabled
        assert origin_amp._enabled == enabled
        assert amp.low_prec_dtype == low
        assert origin_amp._low_prec_dtype == low
        assert amp.high_prec_dtype == high
        assert origin_amp._high_prec_dtype == high

    origin_enabled = amp.enabled
    origin_high = amp.high_prec_dtype
    origin_low = amp.low_prec_dtype
    with amp.autocast(low_prec_dtype="low", high_prec_dtype="high"):
        check(True, "low", "high")
    check(origin_enabled, origin_low, origin_high)
    amp.enabled = True
    amp.high_prec_dtype = "high"
    amp.low_prec_dtype = "low"
    check(True, "low", "high")
    amp.enabled = origin_enabled
    amp.high_prec_dtype = origin_high
    amp.low_prec_dtype = origin_low
    check(origin_enabled, origin_low, origin_high)
