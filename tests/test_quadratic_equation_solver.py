import pytest
import numpy as np
from src.quadratic_equation_solver import standard_formula, alternative_formula, stable_formula

# 设置浮点数比较的相对误差容限
REL_TOL = 1e-10

def test_standard_case():
    """测试标准情况：a、b、c都不为0的情况"""
    # x^2 + 2x + 1 = 0，有一个重根 x = -1
    assert standard_formula(1, 2, 1) == (-1, -1)
    assert alternative_formula(1, 2, 1) == (-1, -1)
    assert stable_formula(1, 2, 1) == (-1, -1)
    
    # x^2 - 5x + 6 = 0，有两个不同的根 x = 2, 3
    x1, x2 = standard_formula(1, -5, 6)
    assert {x1, x2} == {2, 3}
    x1, x2 = alternative_formula(1, -5, 6)
    assert {x1, x2} == {2, 3}
    x1, x2 = stable_formula(1, -5, 6)
    assert {x1, x2} == {2, 3}

def test_no_real_roots():
    """测试无实根的情况（判别式小于0）"""
    # x^2 + x + 1 = 0，无实根
    assert standard_formula(1, 1, 1) is None
    assert alternative_formula(1, 1, 1) is None
    assert stable_formula(1, 1, 1) is None

def test_linear_equation():
    """测试一次方程的情况（a=0）"""
    # 2x + 1 = 0，一次方程
    x1, x2 = stable_formula(0, 2, 1)
    assert x1 == x2 == -0.5
    
    # 0x + 0 = 0，无穷多解
    x1, x2 = stable_formula(0, 0, 0)
    assert x1 == x2 == 0
    
    # 0x + 1 = 0，无解
    assert stable_formula(0, 0, 1) is None

def test_numerical_stability():
    """测试数值稳定性问题"""
    # 0.001x^2 + 1000x + 0.001 = 0
    # 这个方程的根相差很大，可能导致数值不稳定
    a, b, c = 0.001, 1000, 0.001
    
    # 使用标准公式
    x1, x2 = standard_formula(a, b, c)
    # 验证这些根确实满足原方程
    assert abs(a * x1 * x1 + b * x1 + c) < REL_TOL
    assert abs(a * x2 * x2 + b * x2 + c) < REL_TOL
    
    # 使用替代公式
    x1, x2 = alternative_formula(a, b, c)
    assert abs(a * x1 * x1 + b * x1 + c) < REL_TOL
    assert abs(a * x2 * x2 + b * x2 + c) < REL_TOL
    
    # 使用稳定的求根程序
    x1, x2 = stable_formula(a, b, c)
    assert abs(a * x1 * x1 + b * x1 + c) < REL_TOL
    assert abs(a * x2 * x2 + b * x2 + c) < REL_TOL

def test_small_coefficients():
    """测试系数很小的情况"""
    # 0.0001x^2 + 0.0001x + 0.0001 = 0
    x1, x2 = stable_formula(0.0001, 0.0001, 0.0001)
    assert abs(0.0001 * x1 * x1 + 0.0001 * x1 + 0.0001) < REL_TOL
    assert abs(0.0001 * x2 * x2 + 0.0001 * x2 + 0.0001) < REL_TOL