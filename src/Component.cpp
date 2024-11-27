#include "Component.h"

void Component::init()
{
	int n_curves = I.rows();
	if (I(0, 0) == I(n_curves - 1, 3))
		is_loop = true;

	if (is_loop)
	{
		Vector2d v1 = (P.row(I(n_curves - 1, 3)) - P.row(I(n_curves - 1, 2))).normalized();
		Vector2d v2 = (P.row(I(0, 1)) - P.row(I(0, 0))).normalized();
		double theta = std::acos(v1.dot(v2));
		if (theta > G1_tol)
			non_G1.push_back(0);
	}
	else
		non_G1.push_back(0);

	for (int i = 1; i < n_curves; i++)
	{
		Vector2d v1 = (P.row(I(i - 1, 3)) - P.row(I(i - 1, 2))).normalized();
		Vector2d v2 = (P.row(I(i, 1)) - P.row(I(i, 0))).normalized();
		double theta = std::acos(v1.dot(v2));
		if (theta > G1_tol)
			non_G1.push_back(i);
	}
}

