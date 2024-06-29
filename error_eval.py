import numpy as np
from numpy.polynomial.polynomial import Polynomial

def integrate_between_roots(polynomial, x1, x2):
    """
    Compute the definite integral of a polynomial between two of its roots using np.polynomial.Polynomial.integ().
    """
    roots = polynomial.roots()
    roots = np.real(roots[np.isreal(roots)])
    roots = np.sort(roots)

    if roots.size < 2:
        return 0.0

    if x1 is None:
        x1 = roots[0]

    if x2 is None:
        x2 = roots[-1]

    if x2 < x1:
        x1, x2 = x2, x1

    if x2 < roots[0] or x1 > roots[-1]:
        return 0.0

    x1 = max(x1, roots[0])
    x2 = min(x2, roots[-1])

    sign_changes = np.sign(polynomial(roots[:-1])) * np.sign(polynomial(roots[1:])) < 0
    positive_roots = roots[:-1][sign_changes]
    negative_roots = roots[1:][sign_changes]

    if x2 < positive_roots[0]:
        return 0.0

    if x1 > negative_roots[-1]:
        return 0.0

    if x1 < positive_roots[0] and x2 > negative_roots[-1]:
        return polynomial.integ()(negative_roots[-1]) - polynomial.integ()(positive_roots[0])

    integral = 0.0

    if x1 >= positive_roots[0]:
        signs = np.sign(polynomial.deriv()(positive_roots))
        integral += np.sum(np.abs(polynomial.integ()(positive_roots[signs > 0])))

    if x2 <= negative_roots[-1]:
        signs = np.sign(polynomial.deriv()(negative_roots))
        integral += np.sum(np.abs(polynomial.integ()(negative_roots[signs < 0])))

    if x1 < negative_roots[-1] and x2 > positive_roots[0]:
        mask = (positive_roots >= x1) & (positive_roots <= x2)
        signs = np.sign(polynomial.deriv()(positive_roots[mask]))
        integral += np.sum(np.abs(polynomial.integ()(positive_roots[mask][signs > 0])))

        mask = (negative_roots >= x1) & (negative_roots <= x2)
        signs = np.sign(polynomial.deriv()(negative_roots[mask]))
        integral += np.sum(np.abs(polynomial.integ()(negative_roots[mask][signs < 0])))

    return integral

def compute_limit_exceedance_error(p: Polynomial, lb: float, ub: float) -> float:
    """
    Computes the error due to limit exceedance for a given polynomial p and upper and lower bounds (lb, ub).
    """

    # Subtract the linear bounds from the polynomial
    p_ub = Polynomial([1, -ub])
    p_lb = Polynomial([1, -lb])
    p_diff = p - p_lb
    p_diff -= p_ub

    # Find the roots of the difference polynomial within the bounds
    roots = p_diff.roots()
    roots = [root for root in roots if root.imag == 0 and lb <= root.real <= ub]
    roots.sort()

    # Compute the error due to limit exceedance
    error = 0.0
    if len(roots) > 0:
        if p_diff(roots[0]) > 0:
            sign = 1.0
        else:
            sign = -1.0

        for i in range(len(roots) - 1):
            x0 = roots[i]
            x1 = roots[i+1]
            if p_diff(x1) * sign > 0:
                area = np.abs(p_diff.integ()(x1) - p_diff.integ()(x0))
                error += sign * area

            sign *= -1.0

    return error

def test_compute_limit_exceedance_error():
    # Define polynomial and limits
    p = Polynomial([-1, 2, -3, 4, -5])  # Example polynomial
    upper_limit = 2.5
    lower_limit = -1.5

    # Test upper limit exceedance
    u_exceedance = 0.5
    t_exceedance = 2.0
    u = Polynomial([0, 0, u_exceedance])
    x = Polynomial([-1, 1])
    t = np.linspace(0, t_exceedance, num=100)
    error = compute_limit_exceedance_error(p, upper_limit, u, x, t)
    expected_error = u_exceedance * integrate_between_roots(p - upper_limit, x.roots(), t)
    np.testing.assert_almost_equal(error, expected_error)

    # Test lower limit exceedance
    l_exceedance = -1.2
    t_exceedance = 1.0
    u = Polynomial([0, 0, l_exceedance])
    x = Polynomial([1, -1])
    t = np.linspace(0, t_exceedance, num=100)
    error = compute_limit_exceedance_error(p, lower_limit, u, x, t)
    expected_error = np.abs(l_exceedance) * integrate_between_roots(p - lower_limit, x.roots(), t)
    np.testing.assert_almost_equal(error, expected_error)

def test_integrate_between_roots():
    # Define polynomial and roots
    p = Polynomial([1, 0, -2, 0, 3, 0, -4])  # Example polynomial
    roots = [-1, 0, 1.5, 2.3]

    # Test integration between roots where y > 0
    t = np.linspace(0, 1, num=100)
    integral = integrate_between_roots(p, roots[:2], t)
    expected_integral = np.trapz(p(t)[(t >= roots[0]) & (t <= roots[1])], t[(t >= roots[0]) & (t <= roots[1])])
    np.testing.assert_almost_equal(integral, expected_integral)

    # Test integration between roots where y < 0
    t = np.linspace(0, 1, num=100)
    integral = integrate_between_roots(p, roots[2:], t)
    expected_integral = np.trapz(p(t)[(t >= roots[2]) & (t <= roots[3])], t[(t >= roots[2]) & (t <= roots[3])])
    np.testing.assert_almost_equal(integral, expected_integral)
    

if __name__ == '__main__':
    test_compute_limit_exceedance_error()