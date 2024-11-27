#include "Simplify4to3.h"
#include <mutex>
#include"testFun.h"

#define inspect(x) std::cout << std::boolalpha << __LINE__ << ": " #x "=" << (x) << std::endl

void Simplify4to3::init()
{
	int K = components.size();
	imap = VectorXi::Zero(K + 1);
	ids.resize(K);
	exist_list.resize(K);

	int count = 0;
	for (int k = 0; k < K; k++)
	{
		if (components[k].I.rows() < 2 || (components[k].I.rows() == 2 && components[k].is_loop))
		{
			ids[k] = MatrixXi();
			exist_list[k] = VectorXi();
			continue;
		}
		int Nk = 0;
		if (components[k].is_loop)
		{
			Nk = components[k].I.rows();
			MatrixXi temp(Nk, 4);
			VectorXi vec = VectorXi::LinSpaced(Nk, 0, Nk - 1);
			if (Nk >= 4)
			{
				temp << vec.array() - 2, vec.array() - 1, vec, vec.array() + 1;
				temp(0, 0) = Nk - 2;
				temp(0, 1) = temp(1, 0) = Nk - 1;
				temp(Nk - 1, 3) = 0;
			}
			else if (Nk == 3)
			{
				temp << vec.array() - 1, vec, vec.array() + 1, VectorXi::Zero(Nk).array() - 1;
				temp(0, 0) = Nk - 1;
				temp(Nk - 1, 2) = 0;
			}
			else {
				throw std::runtime_error("Not supported.");
			}
			ids[k] = temp;
		}
		else
		{
			Nk = components[k].I.rows() - 3;
			if (Nk >= 1)
			{
				MatrixXi temp(Nk, 4);
				VectorXi vec = VectorXi::LinSpaced(Nk, 0, Nk - 1);
				temp << vec, vec.array() + 1, vec.array() + 2, vec.array() + 3;
				// inspect(temp);
				ids[k] = temp;
			}
			else if (Nk == 0)
			{
				MatrixXi temp(1, 4);
				temp << 0, 1, 2, -1;
				ids[k] = temp;
			}
			else
			{
				MatrixXi temp(1, 4);
				temp << 0, 1, -1, -1;
				ids[k] = temp;
			}
			Nk = max(Nk, 1);
		}
		count += Nk;
		imap(k + 1) = count;
		exist_list[k] = VectorXi::Ones(Nk);
	}
}


struct SegementPair
{
	int gid;
	double err;
	int key = 0;

	SegementPair(int gid, double e) :gid(gid), err(e) {};

	bool operator<(const SegementPair& other) const { return err > other.err; }
};
void Simplify4to3::run()
{
	update_key.assign(imap(imap.size() - 1), 0);
	cached_C.assign(imap(imap.size() - 1), Matrix<double, 10, 2>::Zero());
	priority_queue<SegementPair> Q;
	std::mutex protect_Q;

	#pragma omp parallel for
	for (int k = 0; k < components.size(); k++)
	{
		int Nk = imap(k + 1) - imap(k);
		for (int i = 0; i < Nk; i++)
		{
			pair<Matrix<double, 10, 2>, double> res = compute_cost(i, k, components[k].is_loop);
			int gid = local2global(i, k);
			SegementPair sp(gid, res.second);
			{
				std::lock_guard lock(protect_Q);
				Q.push(sp);
			}
			cached_C[gid] = res.first;
		}
	}

	int min_len = 0;
	for (auto& com : components)
	{
		if (!com.is_loop)
			min_len++;
	}
	int num_collapse = 0;
	while (Q.size() >= min_len)
	{
		SegementPair sp = Q.top();
		Q.pop();
		double cost = sp.err;
		int index = sp.gid;
		pair<int, int> res = global2local(index);
		int lid = res.first, k = res.second;
		if (sp.key != update_key[index] || (exist_list[k].sum() < 2 && components[k].is_loop) || !exist_list[k](lid))
			continue;
		if (cost < fit_tol * 2 && num_collapse < target_num_collapse)
		{
			do_collapse(index);
			num_collapse++;

			if (!components[k].is_loop && exist_list[k].sum() == 1)
			{
				int count = count_if(ids[k].row(lid).data(), ids[k].row(lid).data() + ids[k].row(lid).size(), [](int val) {return val > -1; });
				if (count == 4)
					ids[k].row(lid) = Vector4i(ids[k](lid, 0), ids[k](lid, 1), ids[k](lid, 3), -1);
				else if (count == 3)
					ids[k].row(lid) = Vector4i(ids[k](lid, 0), ids[k](lid, 2), -1, -1);
				else if (count == 2)
					ids[k].row(lid) = Vector4i(ids[k](lid, 0), -1, -1, -1);
				auto [C, err] = compute_cost(lid, k, false);
				update_key[index]++;
				SegementPair sp0(index, err);
				sp0.key = update_key[index];
				Q.push(sp0);
				cached_C[index] = C;
			}
			else
			{
				exist_list[k](lid) = 0;
				if (components[k].is_loop && exist_list[k].sum() < 4)
					update_ids_after_collapse_close_degenerate(lid, k, exist_list[k]);
				else
					update_ids_after_collapse(lid, k, exist_list[k], components[k].is_loop);
				vector<int> nid = get_neighbors(lid, exist_list[k], components[k].is_loop);
				for (int j : nid)
				{
					pair<Matrix<double, 10, 2>, double> res0 = compute_cost(j, k, components[k].is_loop);
					int gid = local2global(j, k);
					update_key[gid]++;
					SegementPair sp0(gid, res0.second);
					sp0.key = update_key[gid];
					Q.push(sp0);
					cached_C[gid] = res0.first;
				}
			}
		}
		else
			break;
	}
	cout << "4->3 number of collapse:" << num_collapse << endl;
	/*writeVectorOfMatricesToTxt(ids, "../simp4to3ids.txt");
	writeSumsOfVectorsToTxt(exist_list, "../simp4to3exist.txt");
	cout << " write success" << endl;*/


	for (int k = 0; k < components.size(); k++)
	{
		MatrixXi Ck = components[k].I;
		MatrixXd PTk = components[k].P;
		vector<int> ng1k = components[k].non_G1;

		Eigen::Array<bool, Eigen::Dynamic, 1> not_all_one = !(Ck.array() == 1).rowwise().all();
		vector<int> Ck_to_keep;
		for (int i = 0; i < not_all_one.size(); ++i) {
			if (not_all_one[i]) {
				Ck_to_keep.push_back(i);
			}
		}
		vector<int> ng1k_new;
		for (int val : ng1k)
		{
			auto it = find(Ck_to_keep.begin(), Ck_to_keep.end(), val);
			if (it != Ck_to_keep.end())
				ng1k_new.push_back(std::distance(Ck_to_keep.begin(), it));
		}

		MatrixXi newCk(Ck_to_keep.size(), Ck.cols());
		for (size_t i = 0; i < Ck_to_keep.size(); ++i) {
			newCk.row(i) = Ck.row(Ck_to_keep[i]);
		}
		MatrixXd RV;
		MatrixXi IMF;
		Utils::removeUnreferenced(PTk, newCk, RV, IMF);

		components[k].I = IMF;
		components[k].P = RV;
		components[k].non_G1 = ng1k_new;
	}
}


int Simplify4to3::local2global(int i, int k)
{
	return imap(k) + i;
}
pair<int, int> Simplify4to3::global2local(int gid)
{
	int lid = 0, k = 0;
	for (int i = 0; i < imap.size(); i++)
	{
		if (imap(i) > gid)
		{
			k = i - 1;
			break;
		}
	}
	lid = gid - imap(k);
	return { lid,k };
}

void Simplify4to3::do_collapse(int i)
{
	Matrix<double, 10, 2> NP = cached_C[i];
	auto [lid, cid] = global2local(i);

	MatrixXi& Ck = components[cid].I;
	MatrixXd& PTk = components[cid].P;
	vector<int>& ng1k = components[cid].non_G1;

	Vector4i id = ids[cid].row(lid);
	int count = count_if(id.data(), id.data() + id.size(), [](int val) {return val > -1; });
	if (count == 4)
	{
		if (any_of(ng1k.begin(), ng1k.end(), [id](int val) {return val == id(2);}))
		{
			RowVector2d center = PTk.row(Ck(id(2), 0));
			if ((center - NP.row(3)).norm() < 1e-10)
				replace_if(ng1k.begin(), ng1k.end(), [id](int val) {return val == id(2); }, id(1));
			else if ((center - NP.row(6)).norm() < 1e-10)
				replace_if(ng1k.begin(), ng1k.end(), [id](int val) {return val == id(2); }, id(3));
		}
		Ck(id(3), 0) = Ck(id(1), 3);
		PTk({ Ck(id(0),1),Ck(id(0),2), Ck(id(0),3), Ck(id(1),1), Ck(id(1),2), Ck(id(1),3), Ck(id(3),1), Ck(id(3),2) }, all) = NP({ 1,2,3,4,5,6,7,8 }, all);
		Ck.row(id(2)) = Vector4i(1, 1, 1, 1);
	}
	else if (count == 3)
	{
		if (any_of(ng1k.begin(), ng1k.end(), [id](int val) {return val == id(1); }))
			replace_if(ng1k.begin(), ng1k.end(), [id](int val) {return val == id(1); }, id(2));
		Ck(id(2), 0) = Ck(id(0), 3);
		PTk({ Ck(id(0),1),Ck(id(0),2), Ck(id(0),3), Ck(id(2),1), Ck(id(2),2) }, all) = NP({ 1,2,3,4,5 }, all);
		Ck.row(id(1)) = Vector4i(1, 1, 1, 1);
	}
	else if (count == 2)
	{
		Ck(id(0), 3) = Ck(id(1), 3);
		PTk({ Ck(id(0),1),Ck(id(0),2) }, all) = NP({ 1,2 }, all);
		Ck.row(id(1)) = Vector4i(1, 1, 1, 1);
	}
	else
	{
		cerr << "Ids should have at least two curves for this collapse" << endl;
	}
}

vector<int> Simplify4to3::get_neighbors(int i, const VectorXi& list, bool isloop)
{
	vector<int> nid;
	vector<int> validList;

	for (int j = 0; j < list.size(); ++j)
	{
		if (list(j) != 0)
		{
			validList.push_back(j);
		}
	}

	if (validList.size() < 1)
	{
		return nid;
	}

	if (validList.size() <= 3)
	{
		nid = validList;
		return nid;
	}

	if (isloop && validList.size() <= 6)
	{
		nid = validList;
		return nid;
	}

	std::vector<int> id1;
	for (int j = validList.size()-1; j >= 0 && id1.size() < 3; --j) {
		if (validList[j] < i) {
			id1.push_back(j);
		}
	}
	std::reverse(id1.begin(), id1.end());
	if (id1.size() < 3)
	{
		for (int j = 0; j < id1.size(); j++)
		{
			nid.push_back(validList[id1[j]]);
		}
		if (isloop)
		{
			for (int j = validList.size()-3+id1.size(); j < validList.size(); j++)
			{
				nid.push_back(validList[j]);
			}
		}
	}
	else
	{
		for (int j = 0; j < id1.size(); j++)
		{
			nid.push_back(validList[id1[j]]);
		}
	}
	

	std::vector<int> id2;
	for (int j = 0; j < validList.size() && id2.size() < 3; ++j) {
		if (validList[j] > i) {
			id2.push_back(j);
		}
	}
	if (id2.size() < 3)
	{
		for (int j = 0; j < id2.size(); j++)
		{
			nid.push_back(validList[id2[j]]);
		}
		if (isloop)
		{
			for (int j = 0; j < 4 - id2.size(); j++)
			{
				nid.push_back(validList[j]);
			}
		}
	}
	else
	{
		for (int j = 0; j < id2.size(); j++)
		{
			nid.push_back(validList[id2[j]]);
		}
	}

	return nid;
}

void Simplify4to3::update_ids_after_collapse(int i, int k, const VectorXi& list, bool isloop)
{
	int deleted_id = ids[k](i, 2);
	ids[k] = (ids[k].array() == deleted_id).select(ids[k](i, 1), ids[k]);
	vector<int> validList;
	for (int j = 0; j < list.size(); ++j)
	{
		if (list(j) != 0)
		{
			validList.push_back(j);
		}
	}

	int id1 = Utils::findIDlast(validList, i);
	if (id1 == -1)
	{
		if (isloop)
		{
			ids[k](validList[validList.size() - 1], 3) = ids[k](i, 3);
		}
	}
	else
	{
		ids[k](validList[id1], 3) = ids[k](i, 3);
	}

	int id2 = Utils::findIDfirst(validList, i);
	if (id2 == -1)
	{
		if (isloop)
		{
			ids[k](validList[0], 0) = ids[k](i, 0);
		}
	}
	else
	{
		ids[k](validList[id2], 0) = ids[k](i, 0);
	}
}

void Simplify4to3::update_ids_after_collapse_close_degenerate(int i, int k, const VectorXi& list)
{
	int count = list.sum();
	MatrixXi& id = ids[k];

	vector<int> validList;
	for (int j = 0; j < list.size(); ++j)
	{
		if (list(j) != 0)
		{
			validList.push_back(j);
		}
	}

	if (count == 3)
	{
		int deleted_id = id(i, 2);
		id = (id.array() == deleted_id).select(id(i, 1), id);

		int id1 = Utils::findIDlast(validList, i);
		if (id1 == -1)
			id(validList[validList.size() - 1], 3) = id(i, 3);
		else
			id(validList[id1], 3) = id(i, 3);

		int id2 = Utils::findIDfirst(validList, i);
		if (id2 == -1)
			id(validList[0], 0) = id(i, 0);
		else
			id(validList[id2], 0) = id(i, 0);

		for (int j : validList)
		{
			id.row(j) = Vector4i(id(j, 1), id(j, 2), id(j, 3), -1);
		}
	}
	else if (count == 2)
	{
		int deleted_id = id(i, 1);
		id = (id.array() == deleted_id).select(id(i, 0), id);
		for (int j : validList)
		{
			id.row(j) = Vector4i(id(j, 0), id(j, 1), -1, -1);
		}
	}
	else
	{
		cerr << "Unexpected bug!" << endl;
	}
}

pair<Matrix<double, 10, 2>, double> Simplify4to3::compute_cost(int i, int k, bool isloop) const
{
	/*inspect(i);
	inspect(k);
	inspect(isloop);*/
	Matrix<double, 10, 2> C = Matrix<double, 10, 2>::Zero();
	double err = INFINITY;

	MatrixXi Ck = components[k].I;
	MatrixXd PTk = components[k].P;
	vector<int> ng1k = components[k].non_G1;
	Vector4i Ci = ids[k].row(i);

	/*inspect(ids.size());
	inspect(ids.data());
	inspect(ids[k]);
	inspect(Ci.transpose());
	inspect(this);*/

	int count = count_if(Ci.data(), Ci.data() + Ci.size(), [](int val) {return val > -1; });
	bool a = any_of(ng1k.begin(), ng1k.end(), [Ci](int val) {return val == Ci(1); });
	bool b = any_of(ng1k.begin(), ng1k.end(), [Ci](int val) {return val == Ci(2); });
	bool c = any_of(ng1k.begin(), ng1k.end(), [Ci](int val) {return val == Ci(3); });
	if (count == 4)
	{
		Matrix<double, 4, 2> C1 = PTk(Ck.row(Ci(0)).array(), all);
		Matrix<double, 4, 2> C2 = PTk(Ck.row(Ci(1)).array(), all);
		Matrix<double, 4, 2> C3 = PTk(Ck.row(Ci(2)).array(), all);
		Matrix<double, 4, 2> C4 = PTk(Ck.row(Ci(3)).array(), all);
		if (a && b && c) {}
		else if (a && b)
		{
			auto [tempC, E, _] = CubicVertexRemovalC0(C3, C4, "C0G1").run();
			C << C1({ 0,1,2 }, all), C2({ 0,1,2 }, all), tempC;
			err = E;
		}
		else if (b && c)
		{
			auto [tempC, E, _] = CubicVertexRemovalC0(C1, C2, "G1C0").run();
			C << tempC, C3({ 1,2,3 }, all), C4({ 1,2,3 }, all);
			err = E;
		}
		else if (b)
		{
			auto [Cl, errl, _0] = CubicVertexRemovalC0(C1, C2, "G1C0").run();
			auto [Cr, errr, _1] = CubicVertexRemovalC0(C3, C4, "C0G1").run();
			if (errl < errr)
			{
				C << Cl, C3({ 1,2,3 }, all), C4({ 1,2,3 }, all);
				err = errl;
			}
			else
			{
				C << C1({ 0,1,2 }, all), C2({ 0,1,2 }, all), Cr;
				err = errr;
			}
		}
		else if (a)
		{
			auto [tempC, E] = KdCurves3to2(C2, C3, C4, false).run();
			C << C1({ 0,1,2 }, all), tempC;
			err = E;
		}
		else if (c)
		{
			auto [tempC, E] = KdCurves3to2(C1, C2, C3, false).run();
			C << tempC, C4({ 1,2,3 }, all);
			err = E;
		}
		else
		{
			if (isloop && (C1.row(0)-C4.row(3)).norm() < 1e-10 && !any_of(ng1k.begin(), ng1k.end(), [Ci](int val) {return val == Ci(0); }))
			{
				auto [tempC, E] = KdCurves4to3(C1, C2, C3, C4, false).run();       //??????????????????????
				C = tempC;
				err = E;
			}
			else
			{
				auto [tempC, E] = KdCurves4to3(C1, C2, C3, C4, false).run();
				C = tempC;
				err = E;
			}
		}

	}
	else if (count == 3)
	{
		if (a && b) {}
		else
		{
			Matrix<double, 4, 2> C1 = PTk(Ck.row(Ci(0)).array(), all);
			Matrix<double, 4, 2> C2 = PTk(Ck.row(Ci(1)).array(), all);
			Matrix<double, 4, 2> C3 = PTk(Ck.row(Ci(2)).array(), all);
			if (a)
			{
				auto [tempC, E, _] = CubicVertexRemovalC0(C2, C3, "C0G1").run();
				C << C1({ 0,1,2 }, all), tempC, MatrixXd::Zero(3, 2);
				err = E;
			}
			else if (b)
			{
				auto [tempC, E, _] = CubicVertexRemovalC0(C1, C2, "G1C0").run();
				C << tempC, C3({ 1,2,3 }, all), MatrixXd::Zero(3, 2);
				err = E;
			}
			else
			{
				if (isloop && !any_of(ng1k.begin(), ng1k.end(), [Ci](int val) {return val == Ci(0); }))
				{
					auto [tempC, E] = KdCurves3to2(C1, C2, C3, false).run(); //??????????????????????????
					C << tempC, MatrixXd::Zero(3, 2);
					err = E;
				}
				else
				{
					auto [tempC, E] = KdCurves3to2(C1, C2, C3, false).run(); //??????????????????????????
					C << tempC, MatrixXd::Zero(3, 2);
					err = E;
				}
			}
		}
	}
	else if (count == 2)
	{
		if (a) {}
		else
		{
			Matrix<double, 4, 2> C1 = PTk(Ck.row(Ci(0)).array(), all);
			Matrix<double, 4, 2> C2 = PTk(Ck.row(Ci(1)).array(), all);
			auto [tempC, E, _] = CubicVertexRemovalC0(C1, C2, "G1G1").run();
			C << tempC, MatrixXd::Zero(6, 2);
			err = E;
		}
	}
	return { C,err };
}