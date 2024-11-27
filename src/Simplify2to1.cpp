#include "Simplify2to1.h"
#include <mutex>
#include"testFun.h"
#define inspect(x) std::cout << std::boolalpha << __LINE__ << ": " #x "=" << (x) << std::endl

void Simplify2to1::init()
{
	int K = components.size();
	imap = VectorXi::Zero(K + 1);
	ids.resize(K);
	exist_list.resize(K);

	int count = 0;
	for (int i = 0; i < K; i++)
	{
		if (components[i].I.rows() < 2)
		{
			ids[i] = MatrixXi();
			exist_list[i] = VectorXi();
			continue;
		}
		int Nk = 0;
		if (components[i].is_loop)
		{
			Nk = components[i].I.rows();
			MatrixXi temp(Nk, 2);
			temp << VectorXi::LinSpaced(Nk, 0, Nk - 1).array() - 1, VectorXi::LinSpaced(Nk, 0, Nk - 1);
			ids[i] = temp;
			ids[i](0, 0) = Nk - 1;
		}
		else
		{
			Nk = components[i].I.rows() - 1;
			MatrixXi temp(Nk, 2);
			temp << VectorXi::LinSpaced(Nk, 0, Nk - 1), VectorXi::LinSpaced(Nk, 0, Nk - 1).array() + 1;
			ids[i] = temp;
		}
		count += Nk;
		imap(i + 1) = count;
		exist_list[i] = VectorXi::Ones(Nk);
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
void Simplify2to1::run()
{
	omp_set_num_threads(20);
	update_key.assign(imap(imap.size() - 1), 0);
	cached_C.assign(imap(imap.size() - 1), Matrix<double, 4, 2>::Zero());
	priority_queue<SegementPair> Q;
	std::mutex protect_Q;

	#pragma omp parallel for
	for (int k = 0; k < components.size(); k++)
	{
		int Nk = imap(k + 1) - imap(k);
		for (int i = 0; i < Nk; i++)
		{
			pair<Matrix<double, 4, 2>, double> res = compute_cost(i, k, components[k].is_loop);
			int gid = local2global(i, k);
			SegementPair sp(gid, res.second);
			{
				std::lock_guard lock(protect_Q);
				Q.push(sp);
			}
			cached_C[gid] = res.first;
		}
	}


	int min_len = components.size();
	for (auto& com : components)
	{
		if (com.is_loop)
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
		if (sp.key != update_key[index] || exist_list[k].sum() < 2)
			continue;
		if (cost < fit_tol && num_collapse < target_num_collapse)
		{
			do_collapse(index);
			num_collapse++;
			int deleted_id = ids[k](lid, 1);
			ids[k] = (ids[k].array() == deleted_id).select(ids[k](lid, 0), ids[k]);
			exist_list[k](lid) = 0;

			vector<int> nid = get_neighbors(lid, exist_list[k], components[k].is_loop);
			for (int j : nid)
			{
				pair<Matrix<double, 4, 2>, double> res0 = compute_cost(j, k, components[k].is_loop);
				int gid = local2global(j, k);
				update_key[gid]++;
				SegementPair sp(gid, res0.second);
				sp.key = update_key[gid];
				Q.push(sp);
				cached_C[gid] = res0.first;
			}
		}
		else
			break;
	}
	cout << "2->1 number of collapse:" << num_collapse << endl;
	/*writeVectorOfMatricesToTxt(ids, "../simp2to1ids.txt");
	writeSumsOfVectorsToTxt(exist_list, "../simp2to1exist.txt");
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


int Simplify2to1::local2global(int i, int k)
{
	return imap(k) + i;
}

pair<int, int> Simplify2to1::global2local(int gid)
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

void Simplify2to1::do_collapse(int i)
{
	Matrix<double, 4, 2> NP = cached_C[i];
	pair<int, int> res = global2local(i);
	int lid = res.first, cid = res.second;
	Vector2i id = ids[cid].row(lid);
	MatrixXi& tempI = components[cid].I;
	MatrixXd& tempP = components[cid].P;
	tempI(id(0), 3) = tempI(id(1), 3);
	tempP.row(tempI(id(0), 1)) = NP.row(1);
	tempP.row(tempI(id(0), 2)) = NP.row(2);
	tempI.row(id(1)) = VectorXi::Ones(4);
}

vector<int> Simplify2to1::get_neighbors(int i, const VectorXi& list, bool isloop)
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

	if (validList.size() == 1)
	{
		nid.push_back(validList[0]);
		return nid;
	}

	int id1 = Utils::findIDlast(validList, i);
	if (id1 == -1)
	{
		if (isloop)
			nid.push_back(validList.back());
	}
	else
	{
		nid.push_back(validList[id1]);
	}

	int id2 = Utils::findIDfirst(validList, i);
	if (id2 == -1)
	{
		if (isloop)
			nid.push_back(validList.front());
	}
	else
	{
		nid.push_back(validList[id2]);
	}

	return nid;
}

pair<int, int> Simplify2to1::getLeftRightNeighbor(int i, const VectorXi& list, bool isloop)
{
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
		return { -1, -1 };
	}

	if (validList.size() == 1)
	{
		return { validList[0], validList[0] };
	}

	int id1 = Utils::findIDlast(validList, i);
	if (id1 == -1)
	{
		if (isloop)
		{
			id1 = validList.back();
		}
	}

	int id2 = Utils::findIDfirst(validList, i);
	if (id2 == -1)
	{
		if (isloop)
		{
			id2 = validList.front();
		}
	}
	return { id1, id2 };
}


pair<Matrix<double, 4, 2>, double> Simplify2to1::compute_cost(int i, int k, bool isloop)
{
	Matrix<double, 4, 2> C = Matrix<double, 4, 2>::Zero();
	double err = INFINITY;

	MatrixXi Ck = components[k].I;
	MatrixXd PTk = components[k].P;
	vector<int> ng1k = components[k].non_G1;
	Vector2i Ci = ids[k].row(i);

	if (find(ng1k.begin(),ng1k.end(),Ci(1)) == ng1k.end())
	{
		Matrix<double, 4, 2> C1 = PTk(Ck.row(Ci(0)).array(), all);
		Matrix<double, 4, 2> C2 = PTk(Ck.row(Ci(1)).array(), all);
		pair<int, int> res = getLeftRightNeighbor(i, exist_list[k], components[k].is_loop);
		int left = res.first, right = res.second;
		if (left < 0 || find(ng1k.begin(), ng1k.end(), ids[k](left,1)) != ng1k.end())
		{
			if (right < 0 || find(ng1k.begin(), ng1k.end(), ids[k](right, 1)) != ng1k.end())
			{
				tuple<Matrix<double, 4, 2>, double, double> res0 = CubicVertexRemovalC0(C1, C2, "C0C0").run();
				C = get<0>(res0);
				err = get<1>(res0);
			}
			else
			{
				tuple<Matrix<double, 4, 2>, double, double> res0 = CubicVertexRemovalC0(C1, C2, "C0G1").run();
				C = get<0>(res0);
				err = get<1>(res0);
			}
		}
		else
		{
			if (right < 0 || find(ng1k.begin(), ng1k.end(), ids[k](right, 1)) != ng1k.end())
			{
				tuple<Matrix<double, 4, 2>, double, double> res0 = CubicVertexRemovalC0(C1, C2, "G1C0").run();
				C = get<0>(res0);
				err = get<1>(res0);
			}
			else if (Utils::removeDuplicateRows(C1).rows() < 4 || Utils::removeDuplicateRows(C2).rows() < 4)
			{
				tuple<Matrix<double, 4, 2>, double, double> res0 = CubicVertexRemovalC0(C1, C2, "G1G1").run();
				C = get<0>(res0);
				err = get<1>(res0);
			}
			else
			{
				pair<Matrix<double, 4, 2>, double> res0 = CubicVertexRemoval(C1, C2).run();
				C = res0.first;
				err = res0.second;
			}
		}
	}
	return { C,err };
}



