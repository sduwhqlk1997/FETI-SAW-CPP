#include <FETI_SAW.hpp>
#include <myOMP.hpp>
#include <stdexcept>
#include <myEigen.hpp>
#include <utility>

namespace FETI
{
#pragma omp declare reduction(matComplexAdd : Mat_t<myFEM::Complex> : omp_out += omp_in) \
    initializer(omp_priv = Mat_t<myFEM::Complex>::Zero(omp_orig.rows(), omp_orig.cols()))
    void SubRegion::genIntInnDofInfo()
    {
        Idx szK = dofIdx.rows();
        Idx nLoLag = bLag2Nodes.size();
        innDofIdx = std::vector<Idx>(szK, -1);
        intDofIdx = std::vector<Idx>(szK, -1);
        for (Idx i = 0; i < nLoLag; ++i)
        {
            Idx idxTemp = bLag2Nodes[i];
            intDofIdx[idxTemp] = 1; // 标记界面自由度
        }
        numIntDof = 0;
        numInnDof = 0;
        for (Idx i = 0; i < szK; ++i)
        {
            if (intDofIdx[i] == 1) // 界面自由度
            {
                intDofIdx[i] = numIntDof; // 界面自由度编号
                numIntDof++;
            }
            else // 内部自由度
            {
                innDofIdx[i] = numInnDof; // 内部自由度编号
                numInnDof++;
            }
        }
    }
    Mat_t<myFEM::Complex> SubRegion::BmulV(const Mat_t<myFEM::Complex> &V, Idx numLag) const
    {
        // Idx numDof = dofIdx.rows();
        Mat_t<myFEM::Complex> res(numLag, V.cols());
        res.setZero();
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx idxLag = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            res.row(idxLag) += sign * V.row(dofIdx1);
        }
        return res;
    }
    Mat_t<myFEM::Complex> SubRegion::BmulV(const RowSparseMat &V, Idx numLag) const
    {
        // 把V转换为行优先
        // using RowSparseMat = Eigen::SparseMatrix<myFEM::Complex, Eigen::RowMajor>;
        // RowSparseMat rowV = V;
        Mat_t<myFEM::Complex> res(numLag, V.cols());
        res.setZero();
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx idxLag = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            res.row(idxLag) += sign * V.row(dofIdx1);
        }
        return res;
    }
    Mat_t<myFEM::Complex> SubRegion::BTmulV(const Mat_t<myFEM::Complex> &V) const
    {
        Idx numDof = dofIdx.rows();
        Mat_t<myFEM::Complex> res(numDof, V.cols());
        res.setZero();
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx lagIdx = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            res.row(dofIdx1) += sign * V.row(lagIdx);
        }
        return res;
    }
    Mat_t<myFEM::Complex> SubRegion::BTmulV(const RowSparseMat &V) const
    {
        Idx numDof = dofIdx.rows();
        Mat_t<myFEM::Complex> res(numDof, V.cols());
        res.setZero();
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx lagIdx = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            res.row(dofIdx1) += sign * V.row(lagIdx);
        }
        return res;
    }
    myFEM::SparseMat_t<myFEM::Complex> SubRegion::BTmulV_sp(const RowSparseMat &V) const
    {
        // using RowSparseMat = Eigen::SparseMatrix<myFEM::Complex, Eigen::RowMajor>;
        // RowSparseMat rowV = V;
        Idx numDof = dofIdx.rows();
        myFEM::SparseMat_t<myFEM::Complex> res(numDof, V.cols());
        std::vector<Eigen::Triplet<myFEM::Complex, Idx>> tripletList;
        tripletList.reserve(V.nonZeros());
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx idxLag = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            for (RowSparseMat::InnerIterator iter(V, idxLag); iter; ++iter)
            {
                tripletList.emplace_back(dofIdx1, iter.col(),
                                         myFEM::Complex(sign, 0) * iter.value());
            }
        }
        res.setFromTriplets(tripletList.begin(), tripletList.end());
        return res;
    }
    Mat_t<myFEM::Complex> SubRegion::BTmulV_int(const Mat_t<myFEM::Complex> &V) const
    {
        Mat_t<myFEM::Complex> res(numIntDof, V.cols());
        res.setZero();
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx lagIdx = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            res.row(intDofIdx[dofIdx1]) +=
                myFEM::Complex(sign, 0) * V.row(lagIdx);
        }
        return res;
    }
    Mat_t<myFEM::Complex> SubRegion::BmulV_int(const Mat_t<myFEM::Complex> &V, Idx numLag) const
    {
        Mat_t<myFEM::Complex> res(numLag, V.cols());
        res.setZero();
        for (Idx i = 0; i < bGlo2Lo.size(); ++i)
        {
            Idx idxLag = bGlo2Lo[i];
            Idx dofIdx1 = bLag2Nodes[i];
            int sign = bSign[i];
            res.row(idxLag) += sign * V.row(intDofIdx[dofIdx1]);
        }
        return res;
    }
    std::vector<SubRegion> initSubRegsFromDevArr(const SAW2_5D::deviceArray &devArr)
    {
        std::vector<SubRegion> subRegs;
        Idx ny = 1;
        Idx nx = devArr.baseStructureArray.cols();
        Idx nz = devArr.baseStructureArray.rows();
        auto idx2D = [nx](Idx idx1D) -> std::pair<Idx, Idx>
        {
            return {idx1D % nx, idx1D / nx};
            // return i + k * nx;
        };
        subRegs.resize(nx * nz);
#pragma omp parallel
        {
            int rank = omp_get_thread_num();
            int ompSize = omp_get_num_threads();
            auto jobList = myOMP::distributeTasks(ompSize, subRegs.size());
            int tStart = jobList[rank].first;
            int tEnd = tStart + jobList[rank].second;
            for (int nSubReg = tStart; nSubReg < tEnd; ++nSubReg)
            {
                auto [i, k] = idx2D(nSubReg);
                Idx probType = devArr.baseStructureArray(k, i);
                subRegs[nSubReg] = SubRegion(devArr.geoArray(k, i).dom_p,
                                             devArr.geoArray(k, i).dom_e,
                                             devArr.subProbs[probType].dofIdx);
                subRegs[nSubReg].probType = probType;
                SAW2_5D::translate3dPoints(subRegs[nSubReg].dofIdx,
                                           devArr.subProbs[probType].dom_p.col(0),
                                           subRegs[nSubReg].dom_p.col(0));
            }
        }
        return subRegs;
    }
    std::vector<std::unordered_map<Vec_t<double>, Idx,
                                   myFEM::PointNdHash, myFEM::PointNdEqual>>
    genSubRegBdIdx(SubRegion &subReg)
    {
        using MapType = std::unordered_map<
            Vec_t<double>,
            Idx,
            myFEM::PointNdHash,
            myFEM::PointNdEqual>;
        Vec_t<double> width = subReg.dom_p.col(1) - subReg.dom_p.col(0);
        double eps = EPS * width.maxCoeff();
        MapType map(
            0,
            myFEM::PointNdHash(eps),
            myFEM::PointNdEqual(eps));

        std::vector<MapType> bdIdxMaps(6, map);
        for (Idx i = 0; i < subReg.dofIdx.rows(); ++i)
        {
            Vec_t<double> pt = subReg.dofIdx.row(i);
            if (myFEM::almostEqual(pt(0), subReg.dom_p(0, 0), eps))
                bdIdxMaps[0][pt] = i; // left
            if (myFEM::almostEqual(pt(0), subReg.dom_p(0, 1), eps))
                bdIdxMaps[1][pt] = i; // right
            if (myFEM::almostEqual(pt(1), subReg.dom_p(1, 1), eps))
                bdIdxMaps[3][pt] = i; // front
            if (myFEM::almostEqual(pt(1), subReg.dom_p(1, 0), eps))
                bdIdxMaps[2][pt] = i; // back
            if (myFEM::almostEqual(pt(2), subReg.dom_p(2, 0), eps))
                bdIdxMaps[4][pt] = i; // bottom
            if (myFEM::almostEqual(pt(2), subReg.dom_p(2, 1), eps))
                bdIdxMaps[5][pt] = i; // top
        }
        return bdIdxMaps;
    }
    std::pair<std::vector<Idx>, std::vector<Idx>> findCommDofOnSingleFace(std::unordered_map<Vec_t<double>, Idx,
                                                                                             myFEM::PointNdHash, myFEM::PointNdEqual>
                                                                              face1,
                                                                          std::unordered_map<Vec_t<double>, Idx,
                                                                                             myFEM::PointNdHash, myFEM::PointNdEqual>
                                                                              face2)
    {
        std::vector<Idx> idx1, idx2;
        idx1.reserve(face1.size());
        idx2.reserve(face2.size());
        for (auto p1 = face1.begin(); p1 != face1.end(); ++p1)
        {
            auto p2 = face2.find(p1->first);
            if (p2 == face2.end())
            {
                std::cout << "点" << p1->first << "在相邻区域上未找到\n";
                std::exit(EXIT_FAILURE);
            }
            else
            {
                idx1.emplace_back(p1->second);
                idx2.emplace_back(p2->second);
            }
        }
        return {idx1, idx2};
    }
    Idx genLagInfo(std::vector<SubRegion> &subRegs, Idx nx, Idx ny, Idx nz)
    {
        std::vector<std::vector<std::unordered_map<Vec_t<double>, Idx,
                                                   myFEM::PointNdHash, myFEM::PointNdEqual>>>
            bdFaceIdx(subRegs.size());
#pragma omp parallel for
        for (Idx i = 0; i < subRegs.size(); ++i)
            bdFaceIdx[i] = genSubRegBdIdx(subRegs[i]);
        Idx numLag = 0;
        auto addLagConnection = [&](Idx leftIdx, Idx rightIdx,
                                    const std::vector<Idx> &leftNodes,
                                    const std::vector<Idx> &rightNodes)
        {
            Idx nLagAdd = static_cast<Idx>(leftNodes.size());
            std::vector<Idx> bGlo2LoAdd(nLagAdd);
            std::iota(bGlo2LoAdd.begin(), bGlo2LoAdd.end(), numLag);
            subRegs[leftIdx].bGlo2Lo.insert(subRegs[leftIdx].bGlo2Lo.end(), bGlo2LoAdd.begin(), bGlo2LoAdd.end());
            subRegs[rightIdx].bGlo2Lo.insert(subRegs[rightIdx].bGlo2Lo.end(), bGlo2LoAdd.begin(), bGlo2LoAdd.end());
            subRegs[leftIdx].bLag2Nodes.insert(subRegs[leftIdx].bLag2Nodes.end(), leftNodes.begin(), leftNodes.end());
            subRegs[rightIdx].bLag2Nodes.insert(subRegs[rightIdx].bLag2Nodes.end(), rightNodes.begin(), rightNodes.end());
            subRegs[leftIdx].bSign.insert(subRegs[leftIdx].bSign.end(), nLagAdd, 1);
            subRegs[rightIdx].bSign.insert(subRegs[rightIdx].bSign.end(), nLagAdd, -1);
            numLag += nLagAdd;
        };
        for (Idx i = 0; i < nx; ++i)
            for (Idx j = 0; j < ny; ++j)
                for (Idx k = 0; k < nz; ++k)
                {
                    Idx thisIdx = idx1D(i, j, k, nx, ny);
                    if (i < nx - 1)
                    {
                        Idx nextIdx = idx1D(i + 1, j, k, nx, ny);
                        auto [idx1, idx2] = findCommDofOnSingleFace(bdFaceIdx[thisIdx][1], bdFaceIdx[nextIdx][0]);
                        addLagConnection(thisIdx, nextIdx, idx1, idx2);
                    }
                    if (j < ny - 1)
                    {
                        Idx nextIdx = idx1D(i, j + 1, k, nx, ny);
                        auto [idx1, idx2] = findCommDofOnSingleFace(bdFaceIdx[thisIdx][3], bdFaceIdx[nextIdx][2]);
                        addLagConnection(thisIdx, nextIdx, idx1, idx2);
                    }
                    if (k < nz - 1)
                    {
                        Idx nextIdx = idx1D(i, j, k + 1, nx, ny);
                        auto [idx1, idx2] = findCommDofOnSingleFace(bdFaceIdx[thisIdx][5], bdFaceIdx[nextIdx][4]);
                        addLagConnection(thisIdx, nextIdx, idx1, idx2);
                    }
                }
#pragma omp parallel for
        for (Idx i = 0; i < subRegs.size(); ++i)
        {
            for (Idx j = 0; j < subRegs[i].bLag2Nodes.size(); ++j)
            {
                Idx nodeIdx = subRegs[i].bLag2Nodes[j];
                subRegs[i].lagType.emplace_back(subRegs[i].dofIdx(nodeIdx, 3));
            }
        }
        return numLag;
    }
    Mat_t<myFEM::Complex> DDMInfo::QmulV(const Mat_t<myFEM::Complex> &V, QSaveType qSaveType, bool ifOMP) const
    {
        // V 的行数应等于粗空间的总维数（各子区域粗基函数数目的总和）。
        // 本函数利用 coaSpa.Q 的分块结构计算 Q*V：每个子区域只处理
        // V 中属于自身的粗空间行，并将结果写回全局拉格朗日乘子空间。
        // ifOMP 控制是否启用 OpenMP 并行，false 时执行串行计算。
        switch (qSaveType)
        {
        case blocks:
        {
            Mat_t<myFEM::Complex> res(nLags, V.cols());
            res.setZero();
#pragma omp parallel for if (ifOMP)
            for (Idx m = 0; m < coaSpa.Q_blk.size(); ++m)
            {
                const Idx nBasis = coaSpa.loDim[m];
                if (nBasis == 0)
                    continue;
                const Idx firstBasis = coaSpa.gloBasIdx[m].first;
                res.middleRows(coaSpa.gloLagIdx[m].first,
                               coaSpa.gloLagIdx[m].second - coaSpa.gloLagIdx[m].first + 1) =
                    coaSpa.Q_blk[m] * V.middleRows(firstBasis, nBasis);
            }
            return res;
            break;
        }
        case full:
        {
            Mat_t<myFEM::Complex> res = coaSpa.Q_full * V;
            return res;
            break;
        }
        default:
            break;
        }
        return Mat_t<myFEM::Complex>(0, 0);
    }
    Mat_t<myFEM::Complex> DDMInfo::QHmulV(const Mat_t<myFEM::Complex> &V, QSaveType qSaveType, bool ifOMP) const
    {
        // V 的行数应等于拉格朗日乘子空间的总维数nLags。
        // 本函数利用 coaSpa.Q 的分块结构计算 Q^H*V：每个子区域只处理
        // V 中属于自身的拉格朗日乘子行，并将结果写回全局粗空间。
        // ifOMP 控制是否启用 OpenMP 并行，false 时执行串行计算。
        switch (qSaveType)
        {
        case blocks:
        {
            Idx nCoarse = 0;
            for (const Idx nBasis : coaSpa.loDim)
                nCoarse += nBasis;
            Mat_t<myFEM::Complex> res(nCoarse, V.cols());
            res.setZero();
#pragma omp parallel for schedule(guided) if (ifOMP)
            for (Idx m = 0; m < coaSpa.Q_blk.size(); ++m)
            {
                const Idx nBasis = coaSpa.loDim[m];
                if (nBasis == 0)
                    continue;
                const Idx firstBasis = coaSpa.gloBasIdx[m].first;
                res.middleRows(firstBasis, nBasis).noalias() =
                    coaSpa.Q_blk[m].adjoint() *
                    V.middleRows(coaSpa.gloLagIdx[m].first,
                                 coaSpa.gloLagIdx[m].second - coaSpa.gloLagIdx[m].first + 1);
            }
            return res;
            break;
        }
        case full:
        {
            Mat_t<myFEM::Complex> res = coaSpa.Q_full.adjoint() * V;
            return res;
            break;
        }
        default:
            break;
        }
        return Mat_t<myFEM::Complex>(0, 0);
    }
    void DDMInfo::initCoarseProblem(std::pair<double, double> ranBAWVec, int nBAWVec,
                                    std::vector<double> vSAW, RunTimeInfo &runTimeInfo, double tol,
                                    QSaveType qSaveType, Idx maxConcurrentAreas, Idx maxBlkSz, bool ifOMP)
    {
        // Idx maxConcurrentAreas 对每类子区域分组计算FQ然后合并，maxConcurrentAreas为每组最大子区域数，若<0则不分组
        // Idx maxBlkSz 对于QSaveType::full，按矩阵Q分块计算QHFQ时的最大块大小，若<0则不分块
        initCoarseSpace(ranBAWVec, nBAWVec, vSAW, qSaveType, tol);
        std::cout << "已创建粗空间，开始形成粗问题" << std::endl;
        // 计算(Q^HFQ)^-1的LU分解
        switch (qSaveType)
        {
        case blocks:
        {
            RowSparseMat Q = myEigen::blkdiag<myFEM::Complex, Eigen::RowMajor>(coaSpa.Q_blk, ifOMP);
            coaSpa.luFactor = FQ(Q, runTimeInfo, maxConcurrentAreas, ifOMP);
            // std::cout << "FQ执行完成..." << std::endl;
            coaSpa.luFactor = QHmulV(coaSpa.luFactor, qSaveType, ifOMP); // Q^HFQ
            // coaSpa.ldltFactor = Q.adjoint() * coaSpa.ldltFactor; // Q^HFQ
            break;
        }
        case full:
        {
            // if (maxBlkSz <= 0)
            //     maxBlkSz = 12 * nBAWVec + 16 * vSAW.size();
            runTimeInfo.distributeRHS.resize(subProbs.nSubProbs.size(), 0);
            runTimeInfo.FQ.resize(subProbs.nSubProbs.size(), 0);
            runTimeInfo.collectRHS.resize(subProbs.nSubProbs.size(), 0);
            coaSpa.luFactor = computeQHFQ(runTimeInfo, maxBlkSz, maxConcurrentAreas, ifOMP);
            break;
        }
        }
        std::cout << "开始对粗空间矩阵(Q^HFQ)进行LU分解,总可用线程数为" << mkl_get_max_threads() << "...\n";
        double t1 = omp_get_wtime();
        coaSpa.luPivots.resize(coaSpa.luFactor.rows());
        Idx nCoarse = coaSpa.luFactor.rows();

        if (nCoarse > 0)
        {
            const MKL_INT info = LAPACKE_zgetrf(
                LAPACK_COL_MAJOR, static_cast<MKL_INT>(nCoarse), static_cast<MKL_INT>(nCoarse),
                reinterpret_cast<lapack_complex_double *>(coaSpa.luFactor.data()),
                static_cast<MKL_INT>(coaSpa.luFactor.outerStride()), coaSpa.luPivots.data());
            if (info != 0)
                throw std::runtime_error("LAPACKE_zgetrf failed");
        }
        double t2 = omp_get_wtime();
        std::cout << "粗空间矩阵(Q^HFQ)的LU分解耗时: " << t2 - t1 << " s\n";
    }
    Mat_t<myFEM::Complex> DDMInfo::solveCoaProb(const Mat_t<myFEM::Complex> &b) const
    {
        const auto &factor = coaSpa.luFactor;
        const Idx n = factor.rows();
        if (n == 0 || factor.cols() != n ||
            static_cast<Idx>(coaSpa.luPivots.size()) != n)
            throw std::logic_error("Coarse-space LU factorization is not initialized");
        if (b.rows() != n)
            throw std::invalid_argument("Coarse-problem RHS row count does not match the factorization");
        if (b.cols() == 0)
            return Mat_t<myFEM::Complex>(n, 0);

        Mat_t<myFEM::Complex> res = b;
        const MKL_INT info = LAPACKE_zgetrs(
            LAPACK_COL_MAJOR, 'N', static_cast<MKL_INT>(n), static_cast<MKL_INT>(res.cols()),
            reinterpret_cast<const lapack_complex_double *>(factor.data()),
            static_cast<MKL_INT>(factor.outerStride()), coaSpa.luPivots.data(),
            reinterpret_cast<lapack_complex_double *>(res.data()),
            static_cast<MKL_INT>(res.outerStride()));
        if (info != 0)
            throw std::runtime_error("LAPACKE_zgetrs failed");
        return res;
    }
    DDMInfo::DDMInfo(const SAW2_5D::deviceArray &devArr, Idx nSeg, int segStrategy)
    {
        // segStrategy分段策略，0：将每类子问题对应的子区域都尽量分为nSeg段，1：将每类子问题分成若干段，且每段的子区域数尽量为nSeg
        if (nSeg < 1)
        {
            std::cout << "DDMInfo::DDMInfo：nSeg要大于等于1\n";
            std::exit(EXIT_FAILURE);
        }
        const int nSizeOMP = omp_get_max_threads();
        Idx nx = devArr.baseStructureArray.cols(),
            ny = 1,
            nz = devArr.baseStructureArray.rows();
        subRegs = initSubRegsFromDevArr(devArr);
        nLags = genLagInfo(subRegs, nx, ny, nz);
        this->w = devArr.w;
        int nTypes = devArr.subProbs.size();
        {
            std::vector<std::vector<Idx>> subProb2ArrIdx(nTypes);
            std::vector<Idx> nSubProbs(nTypes, 0);
            for (int k = 0; k < devArr.baseStructureArray.rows(); ++k)
                for (int i = 0; i < devArr.baseStructureArray.cols(); ++i)
                {
                    Idx idx = idx1D(i, 0, k, nx, ny);
                    int typeSubProb = devArr.baseStructureArray(k, i);
                    nSubProbs[typeSubProb]++;
                    subProb2ArrIdx[typeSubProb].emplace_back(idx);
                }
            std::vector<std::vector<std::pair<int, int>>> jobListPerProb(nTypes); // 每行表示一个子问题类型
            switch (segStrategy)
            {
            case 0:
            {
                for (int i = 0; i < nTypes; ++i)
                    jobListPerProb[i] = myOMP::distributeTasks(nSeg, nSubProbs[i]);
                break;
            }
            case 1:
            {
                for (int i = 0; i < nTypes; ++i)
                    jobListPerProb[i] = myOMP::distributeTasks_byWidth(nSeg, nSubProbs[i]);
                break;
            }
            default:
            {
                std::cout << "DDMInfo::DDMInfo：segStrategy取值错误，只可为0或1\n";
                std::exit(EXIT_FAILURE);
            }
            }
            if (!subProbs.nSubProbs.empty())
                subProbs.nSubProbs.clear();
            if (!subProbs.subProb2ArrIdx.empty())
                subProbs.subProb2ArrIdx.clear();
            if (!subProbs.startIdxSubProbType.empty())
                subProbs.startIdxSubProbType.clear();
            if (!subProbs.subProbType.empty())
                subProbs.subProbType.clear();
            for (int iProb = 0; iProb < nTypes; ++iProb)
            {
                std::vector<std::pair<int, int>> *ptrJ = &(jobListPerProb[iProb]);

                subProbs.startIdxSubProbType.emplace_back(subProbs.nSubProbs.size());
                for (int iS = 0; iS < ptrJ->size(); ++iS)
                {
                    int start = ptrJ->at(iS).first, count = ptrJ->at(iS).second;
                    if (count == 0)
                        break;
                    subProbs.nSubProbs.emplace_back(count);
                    subProbs.subProb2ArrIdx.emplace_back(subProb2ArrIdx[iProb].begin() + start,
                                                         subProb2ArrIdx[iProb].begin() + start + count);
                    subProbs.subProbType.emplace_back(iProb);
                }
            }
            subProbs.nloMklThreads = myOMP::assignThreadCount(nSizeOMP, subProbs.nSubProbs);
            // nloMklThreads = subProbs.nloMklThreads;
        }
        // 保存无量纲化信息
        dimScales = devArr.dimScales;
        // 对所有子问题执行LDLT分解
        Idx nProbs = subProbs.nSubProbs.size();
        subProbs.slovers.resize(nProbs);
        subProbs.RHS.resize(nTypes);
        for (int i = 0; i < nProbs; ++i)
            subProbs.slovers[i] = std::make_unique<Solver>();
        // 初始化右端项
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nTypes; ++i)
            subProbs.RHS[i] = devArr.subProbs[i].F;
        std::vector<int> myNLoMklThreads(nProbs, 1); // 每个子问题至少分配一个MKL线程
        if (nProbs > 0 && nProbs <= static_cast<Idx>(nSizeOMP))
        {
            int rec = nSizeOMP / nProbs,
                rem = nSizeOMP % nProbs;
            for (Idx i = 0; i < nProbs; ++i)
            {
                myNLoMklThreads[i] = rem > 0 ? (rec + 1) : rec;
                rem--;
            }
        }
        std::cout << "开始对" << nProbs << "个子问题进行LDL分解,总可用线程数为" << nSizeOMP << "...\n";
        // 开启嵌套并行
        const int previousLevels = omp_get_max_active_levels();
        const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < nProbs; ++i)
                {
#pragma omp task firstprivate(i)
                    {
                        int tP = subProbs.subProbType[i];
                        int old_nt = mkl_set_num_threads_local(myNLoMklThreads[i]);
                        subProbs.slovers[i]->compute(devArr.subProbs[tP].K);
                        mkl_set_num_threads_local(old_nt);
                    }
                }
#pragma omp taskwait
            }
        }
        // 关闭嵌套并行
        omp_set_max_active_levels(previousLevels);
        mkl_set_dynamic(previousMklDynamic);
    }
    DDMInfo::DDMInfo(const SAW2_5D::deviceArray &devArr,
                     std::vector<SAW2_5D::typeMaterial> &subProbMatTypes, Idx nSeg, int segStrategy)
        : DDMInfo(devArr, nSeg, segStrategy)
    {
        int nTypes = devArr.subProbs.size();
        subProbs.para_p.resize(nTypes);
        std::set<int> sPType;
        for (Idx i = 0; i < devArr.baseStructureArray.rows(); ++i)
            for (Idx j = 0; j < devArr.baseStructureArray.cols(); ++j)
            {
                auto p = sPType.find(devArr.baseStructureArray(i, j));
                if (p == sPType.end())
                {
                    sPType.insert(devArr.baseStructureArray(i, j));
                    subProbs.para_p[devArr.baseStructureArray(i, j)] = SAW2_5D::materialLib(
                        subProbMatTypes[devArr.baseStructureArray(i, j)]);
                }
            }
        // 无量纲化
        for (int i = 0; i < nTypes; ++i)
        {
            subProbs.para_p[i].c /= dimScales.c0;
            subProbs.para_p[i].e /= dimScales.e0;
            subProbs.para_p[i].epcl /= dimScales.epcl0;
            subProbs.para_p[i].rho /= dimScales.rho0;
            subProbs.para_p[i].w /= dimScales.omega0;
            subProbs.para_p[i].lambda /= dimScales.c0;
            subProbs.para_p[i].mu /= dimScales.c0;
        }
    }
    void DDMInfo::computeloSol()
    { // 对每类子区域计算一次K^-1*RHS并存储一份loSol，当RHS为全0向量时直接置零
        const Idx nTypes = subProbs.startIdxSubProbType.size();
        subProbs.loSol.resize(nTypes);
        int ompMaxSize = omp_get_max_threads();
        // for (int i = 0; i < nTypes; ++i)
        //     if (subProbs.nSubProbs[i] > 0)
        //         ++nValidTypes;
        std::vector<int> myNLoMklThreads(nTypes, 1); // 分配给每类子问题分配线程数
        if (nTypes > 0 && nTypes <= static_cast<Idx>(ompMaxSize))
        {
            int rec = ompMaxSize / nTypes,
                rem = ompMaxSize % nTypes;
            for (Idx i = 0; i < nTypes; ++i)
            {
                myNLoMklThreads[i] = rem > 0 ? (rec + 1) : rec;
                rem--;
            }
        }
        const int previousLevels = omp_get_max_active_levels();
        const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < nTypes; ++i)
                {
#pragma omp task firstprivate(i)
                    {
                        int idxP = subProbs.startIdxSubProbType[i]; // 获取当前子区域LDLT分解的存储位置
                        int old_nt = mkl_set_num_threads_local(myNLoMklThreads[i]);
                        subProbs.loSol[i] = subProbs.slovers[idxP]->solve(subProbs.RHS[i]);
                        mkl_set_num_threads_local(old_nt);
                    }
                }
#pragma omp taskwait
            }
        }
        // 关闭嵌套并行
        omp_set_max_active_levels(previousLevels);
        mkl_set_dynamic(previousMklDynamic);
    }
    Vec_t<myFEM::Complex> DDMInfo::computeLagRHS(bool ifOMP) const
    {
        // 计算Lagrange乘子线性方程组的右端项，注意在调运此函数之前一定要先完成各子问题loSol的计算
        std::vector<Vec_t<myFEM::Complex>> contributions(subRegs.size());
#pragma omp parallel for if (ifOMP)
        for (Idx n = 0; n < subRegs.size(); ++n)
        {
            Idx probType = subRegs[n].probType;
            if (subProbs.loSol[probType].size() > 0)
                contributions[n] = subRegs[n].BmulV(subProbs.loSol[probType], nLags);
        }
        Vec_t<myFEM::Complex> res = Vec_t<myFEM::Complex>::Zero(nLags);
        for (Idx n = 0; n < subRegs.size(); ++n)
        {
            if (contributions[n].size() > 0)
                res += contributions[n];
        }
        return res;
    }
    template <typename MatType>
    Mat_t<myFEM::Complex> DDMInfo::distributeRHS(const MatType &V, Idx probType, bool ifOMP) const
    {
        // V为复值稀疏矩阵或者复值稠密矩阵，行数等于lagrange乘子总数
        // probType为子区域对应的子问题片段编号
        // 本函数按subProbs.subProb2ArrIdx[probType]中记录的子区域索引收并行地执行subRegs[i].BTmulV(V)操作 (i\in subProbs.subProb2ArrIdx[probType])
        // 然后将结果按subProbs.subProb2ArrIdx[probType]的索引顺序，将各子区域的计算结果按列依次收集到一个复值稠密矩阵输出
        const auto &arrIdx = subProbs.subProb2ArrIdx[probType];
        // if (arrIdx.empty())
        //     return Mat_t<myFEM::Complex>(0, 0);

        const Idx nDof = subRegs[arrIdx.front()].dofIdx.rows();
        const Idx nCols = V.cols();
        Mat_t<myFEM::Complex> res(nDof, arrIdx.size() * nCols);
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < arrIdx.size(); ++i)
            res.middleCols(i * nCols, nCols) = subRegs[arrIdx[i]].BTmulV(V);
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::collectRHS(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP) const
    {
        // V为复值稠密矩阵，行数等于子区域自由度总数
        // probType为子区域对应的子问题段编号
        // nCol = V.cols()/subProbs.nSubProbs[probType] (我保证一定整除)
        // 将V按nCol列分块，每个子区域对应一块，按subProbs.subProb2ArrIdx[probType]中记录的子区域索,按子区域并行地执行subRegs[i].BmulV(V)操作 (i\in subProbs.subProb2ArrIdx[probType])
        // 最后将各自区域计算结果相加得到输出结果
        const auto &arrIdx = subProbs.subProb2ArrIdx[probType];
        // if (arrIdx.empty())
        //     return Mat_t<myFEM::Complex>(0, 0);

        const Idx nSubProbs = arrIdx.size();
        const Idx nCols = V.cols() / nSubProbs;
        std::vector<Mat_t<myFEM::Complex>> contributions(nSubProbs);
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < nSubProbs; ++i)
            contributions[i] = subRegs[arrIdx[i]].BmulV(V.middleCols(i * nCols, nCols), nLags);

        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(
            contributions[0].rows(), contributions[0].cols());
#pragma omp parallel for reduction(matComplexAdd : res)
        for (Idx i = 0; i < nSubProbs; ++i)
            res += contributions[i];
        return res;
    }
    template <typename MatType>
    Mat_t<myFEM::Complex> DDMInfo::loFy(const MatType &V, Idx probType, bool ifOMP, int nMKL) const
    {
        // V为复值稠密矩阵，行数等于子区域自由度总数
        // probType为子区域对应的子问题段编号
        // ifOMP为是否使用OpenMP并行

        // 设置mkl
        int old_nt = mkl_set_num_threads_local(nMKL);
        Mat_t<myFEM::Complex> res = distributeRHS(V, probType, ifOMP);
        res = subProbs.slovers[probType]->solve(res);
        res = collectRHS(res, probType, ifOMP);
        // 还原mkl
        mkl_set_num_threads_local(old_nt);
        return res;
    }
    template <typename MatType>
    Mat_t<myFEM::Complex> DDMInfo::Fy(const MatType &V, bool ifOMP) const
    {
        // V为复值稠密矩阵，行数等于拉格朗日乘子总数
        // ifOMP为是否使用嵌套OpenMP并行（仅控制OMP+OMP嵌套）
        Idx nSegs = subProbs.nSubProbs.size();
        std::vector<Mat_t<myFEM::Complex>> contributions(nSegs);
        // 开启嵌套并行
        const int previousLevels = omp_get_max_active_levels();
        const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            {
                for (Idx i = 0; i < nSegs; ++i)
                {
#pragma omp task firstprivate(i)
                    {
                        const int previousOmpThreads = omp_get_max_threads();
                        omp_set_num_threads(subProbs.nloMklThreads[i]);
                        contributions[i] = loFy(V, i, ifOMP, subProbs.nloMklThreads[i]);
                        omp_set_num_threads(previousOmpThreads);
                    }
                }
#pragma omp taskwait
            }
        }
        omp_set_max_active_levels(previousLevels);
        mkl_set_dynamic(previousMklDynamic);
        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(
            nLags, V.cols());
#pragma omp parallel for reduction(matComplexAdd : res)
        for (Idx i = 0; i < nSegs; ++i)
            res += contributions[i];
        // 关闭嵌套并行

        return res;
    }
    template <typename MatType>
    Mat_t<myFEM::Complex> DDMInfo::Fy_Serial(const MatType &V) const
    {
        Idx nSegs = subProbs.nSubProbs.size();
        std::vector<Mat_t<myFEM::Complex>> contributions(nSegs);
        for (Idx i = 0; i < nSegs; ++i)
        {
            contributions[i] = loFy(V, i, false, 1);
        }
        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(
            contributions[0].rows(), contributions[0].cols());
        for (Idx i = 0; i < nSegs; ++i)
            res += contributions[i];
        return res;
    }
    void DDMInfo::genAllIntInnDofInfo(bool ifOMP)
    {
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < subRegs.size(); ++i)
            subRegs[i].genIntInnDofInfo();
    }
    void DDMInfo::computeSchurComplement(SchurComp &result,
                                         SubRegion &subReg,
                                         const myFEM::SparseMat_t<myFEM::Complex> &K,
                                         int tMKL)
    {
        // 计算Schur补
        using Iter = myFEM::SparseMat_t<myFEM::Complex>::InnerIterator;
        int old_nt = mkl_set_num_threads_local(tMKL);
        // Idx szK = K.rows();
        /*矩阵初始化*/
        result.Kbb.resize(subReg.numIntDof, subReg.numIntDof);
        myFEM::SparseMat_t<myFEM::Complex> Kii(subReg.numInnDof, subReg.numInnDof);
        result.Kbi.resize(subReg.numIntDof, subReg.numInnDof);
        result.Kib.resize(subReg.numInnDof, subReg.numIntDof);
        /* 三元组初始化*/
        Idx nnz = K.nonZeros();
        std::vector<Eigen::Triplet<myFEM::Complex, Idx>> tptKbb, tptKii, tptKbi, tptKib;
        tptKbb.reserve(nnz), tptKbi.reserve(nnz), tptKib.reserve(nnz), tptKii.reserve(nnz);
        for (Idx col = 0; col < K.outerSize(); ++col)
        {
            for (Iter iter(K, col); iter; ++iter)
            {
                Idx i = iter.row(), j = iter.col();
                if (subReg.intDofIdx[i] != -1 && subReg.intDofIdx[j] != -1) // Kbb
                {
                    tptKbb.emplace_back(subReg.intDofIdx[i], subReg.intDofIdx[j], iter.value());
                }
                else if (subReg.intDofIdx[i] == -1 && subReg.intDofIdx[j] == -1) // Kii
                {
                    tptKii.emplace_back(subReg.innDofIdx[i], subReg.innDofIdx[j], iter.value());
                }
                else if (subReg.intDofIdx[i] != -1 && subReg.intDofIdx[j] == -1) // Kbi
                {
                    tptKbi.emplace_back(subReg.intDofIdx[i], subReg.innDofIdx[j], iter.value());
                }
                else if (subReg.intDofIdx[i] == -1 && subReg.intDofIdx[j] != -1) // Kib
                {
                    tptKib.emplace_back(subReg.innDofIdx[i], subReg.intDofIdx[j], iter.value());
                }
            }
        }
        result.Kbb.setFromTriplets(tptKbb.begin(), tptKbb.end());
        result.Kbi.setFromTriplets(tptKbi.begin(), tptKbi.end());
        result.Kib.setFromTriplets(tptKib.begin(), tptKib.end());
        Kii.setFromTriplets(tptKii.begin(), tptKii.end());
        result.invKii->compute(Kii);
        mkl_set_num_threads_local(old_nt);
    }
    void DDMInfo::initDirPrecond(const SAW2_5D::deviceArray &devArr)
    {
        if (subProbs.nSubProbs.size() == 0)
        {
            std::cout << "DDMInfo::initDirPrecond()：请先用deviceAyyar变量初始化DDMInfo变量\n";
            std::exit(EXIT_FAILURE);
        }
        std::cout << "开始构造Dirichlet 预条件子...\n";
        this->genAllIntInnDofInfo(true);
        int nSegs = subProbs.nSubProbs.size();
        dirPrecond.resize(nSegs);
        // 开启嵌套并行
        int ompMaxSize = omp_get_max_threads();
        std::vector<int> myNLoMklThreads(nSegs, 1); // 分配给每个子问题做LDL分解的线程数
        if (nSegs > 0 && nSegs <= static_cast<Idx>(ompMaxSize))
        {
            int rec = ompMaxSize / nSegs,
                rem = ompMaxSize % nSegs;
            for (Idx i = 0; i < nSegs; ++i)
            {
                myNLoMklThreads[i] = rem > 0 ? (rec + 1) : rec;
                rem--;
            }
        }
        const int previousLevels = omp_get_max_active_levels();
        const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < nSegs; ++i)
                {
#pragma omp task firstprivate(i)
                    {
                        int tP = subProbs.subProbType[i];
                        Idx fstIdx = subProbs.subProb2ArrIdx[i][0];
                        computeSchurComplement(dirPrecond[i],
                                               subRegs[fstIdx],
                                               devArr.subProbs[tP].K,
                                               myNLoMklThreads[i]);
                    }
                }
#pragma omp taskwait
            }
        }
        // 关闭嵌套并行
        omp_set_max_active_levels(previousLevels);
        mkl_set_dynamic(previousMklDynamic);
    }
    Mat_t<myFEM::Complex> DDMInfo::distributeRHS_int(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP) const
    {
        // 类似distributeRHS但使用BTmulV_int方法，V为复值稠密矩阵，行数等于拉格朗日乘子总数
        const auto &arrIdx = subProbs.subProb2ArrIdx[probType];
        if (arrIdx.empty())
            return Mat_t<myFEM::Complex>(0, 0);

        const Idx nIntDof = subRegs[arrIdx.front()].numIntDof;
        const Idx nCols = V.cols();
        Mat_t<myFEM::Complex> res(nIntDof, arrIdx.size() * nCols);
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < arrIdx.size(); ++i)
            res.middleCols(i * nCols, nCols) = subRegs[arrIdx[i]].BTmulV_int(V);
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::collectRHS_int(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP) const
    {
        const auto &arrIdx = subProbs.subProb2ArrIdx[probType];
        if (arrIdx.empty())
            return Mat_t<myFEM::Complex>(0, 0);

        const Idx nSubProbs = arrIdx.size();
        const Idx nCols = V.cols() / nSubProbs;
        std::vector<Mat_t<myFEM::Complex>> contributions(nSubProbs);
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < nSubProbs; ++i)
            contributions[i] = subRegs[arrIdx[i]].BmulV_int(V.middleCols(i * nCols, nCols), nLags);

        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(
            nLags, contributions[0].cols());
#pragma omp parallel for reduction(matComplexAdd : res)
        for (Idx i = 0; i < nSubProbs; ++i)
            res += contributions[i];
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::loMy(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP, int nMKL) const
    {
        int old_nt = mkl_set_num_threads_local(nMKL);
        const SchurComp &dirPre = dirPrecond[probType];
        Mat_t<myFEM::Complex> tempRes = distributeRHS_int(V, probType, ifOMP);
        Mat_t<myFEM::Complex> res = dirPre.Kib * tempRes;
        res = dirPre.invKii->solve(res);
        res = dirPre.Kbi * res;
        tempRes = dirPre.Kbb * tempRes;
        res = tempRes - res;
        res = collectRHS_int(res, probType, ifOMP);
        mkl_set_num_threads_local(old_nt);
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::My(const Mat_t<myFEM::Complex> &V, bool ifOMP) const
    {
        // V为复值稠密矩阵，行数等于拉格朗日乘子总数
        Idx nSegs = subProbs.nSubProbs.size();
        std::vector<Mat_t<myFEM::Complex>> contributions(nSegs);
        const int previousLevels = omp_get_max_active_levels();
        const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            {
                for (Idx i = 0; i < nSegs; ++i)
                {
#pragma omp task firstprivate(i)
                    {
                        const int previousOmpThreads = omp_get_max_threads();
                        omp_set_num_threads(subProbs.nloMklThreads[i]);
                        contributions[i] = loMy(V, i, ifOMP, subProbs.nloMklThreads[i]);
                        omp_set_num_threads(previousOmpThreads);
                    }
                }
#pragma omp taskwait
            }
        }
        omp_set_max_active_levels(previousLevels);
        mkl_set_dynamic(previousMklDynamic);
        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(
            nLags, V.cols());
#pragma omp parallel for reduction(matComplexAdd : res)
        for (Idx i = 0; i < nSegs; ++i)
            res += contributions[i];

        return res;
    }
    DDMInfo::BTQ DDMInfo::computeBTQ(const RowSparseMat &Q,
                                     const std::vector<Idx> &regionSegment, bool ifOMP) const
    {
        Idx nProbs = regionSegment.size();
        // using RowSparseMat = Eigen::SparseMatrix<myFEM::Complex, Eigen::RowMajor, Idx>;
        // const RowSparseMat rowQ = Q;
        std::vector<myFEM::SparseMat_t<myFEM::Complex>> tempBTQ(nProbs);
#pragma omp parallel for if (ifOMP)
        for (Idx subRegionIndex = 0; subRegionIndex < nProbs; ++subRegionIndex)
        {
            const Idx arrIdx = regionSegment[subRegionIndex];
            const SubRegion &subReg = subRegs[arrIdx];
            tempBTQ[subRegionIndex] = subReg.BTmulV_sp(Q);
        }
        BTQ res;
        res.loIdx.resize(nProbs);
        res.segRange.resize(nProbs);
        res.dimQ = Q.cols();
        Idx nNonZeroCols = 0;
        for (Idx i = 0; i < nProbs; ++i)
        {
            res.segRange[i].first = nNonZeroCols;
            for (Idx col = 0; col < tempBTQ[i].outerSize(); ++col)
            {
                if (tempBTQ[i].innerVector(col).nonZeros() > 0)
                {
                    res.loIdx[i].emplace_back(col);
                    nNonZeroCols++;
                }
            }
            res.segRange[i].second = nNonZeroCols - 1;
        }
        res.BTQ.resize(tempBTQ[0].rows(), nNonZeroCols);
        Idx pos = 0;
        // 待优化
        for (Idx i = 0; i < nProbs; ++i)
        {
            for (Idx j = 0; j < res.loIdx[i].size(); ++j)
            {
                res.BTQ.col(pos) = tempBTQ[i].col(res.loIdx[i][j]);
                pos++;
            }
        }
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::computeBABTQ(BTQ &ABTQ,
                                                const std::vector<Idx> &regionSegment,
                                                bool ifOMP) const
    {
        Idx nProbs = regionSegment.size();
        std::vector<Mat_t<myFEM::Complex>> tempBABTQ(nProbs);
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < nProbs; ++i)
        {
            Idx arrIdx = regionSegment[i];
            tempBABTQ[i] = subRegs[arrIdx].BmulV(ABTQ.BTQ.middleCols(ABTQ.segRange[i].first,
                                                                     ABTQ.segRange[i].second -
                                                                         ABTQ.segRange[i].first + 1),
                                                 nLags);
        }
        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(nLags, ABTQ.dimQ);
        std::vector<std::vector<std::pair<Idx, Idx>>> entriesByColumn(ABTQ.dimQ);
        for (Idx i = 0; i < nProbs; ++i)
        {
            for (Idx j = 0; j < ABTQ.loIdx[i].size(); ++j)
                entriesByColumn[ABTQ.loIdx[i][j]].emplace_back(i, j);
        }
#pragma omp parallel for if (ifOMP)
        for (Idx col = 0; col < ABTQ.dimQ; ++col)
            for (const auto &entry : entriesByColumn[col])
                res.col(col) += tempBABTQ[entry.first].col(entry.second);

        // 原始串行累加写法（重新启用时，需替换上面的按列并行累加）：
        // for (Idx i = 0; i < nProbs; ++i)
        // {
        //     for (Idx j = 0; j < ABTQ.loIdx[i].size(); ++j)
        //     {
        //         res.col(ABTQ.loIdx[i][j]) += tempBABTQ[i].col(j);
        //     }
        // }
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::loFQ(const RowSparseMat &Q, RunTimeInfo &runTimeInfo, Idx probType, bool ifOMP,
                                        int nMKL, Idx maxConcurrentAreas) const
    {
        int old_nt = mkl_set_num_threads_local(nMKL);
        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(nLags, Q.cols());
        if (maxConcurrentAreas == -1) // 不分组计算
        {
            double tStart = omp_get_wtime();
            BTQ tempRes = computeBTQ(Q, subProbs.subProb2ArrIdx[probType], ifOMP);
            double tEnd = omp_get_wtime();
            runTimeInfo.distributeRHS[probType] += tEnd - tStart;
            tStart = omp_get_wtime();
            if (tempRes.BTQ.cols() > 0)
                tempRes.BTQ = subProbs.slovers[probType]->solve(tempRes.BTQ);
            tEnd = omp_get_wtime();
            runTimeInfo.FQ[probType] += tEnd - tStart;
            tStart = omp_get_wtime();
            res = computeBABTQ(tempRes, subProbs.subProb2ArrIdx[probType], ifOMP);
            tEnd = omp_get_wtime();
            runTimeInfo.collectRHS[probType] += tEnd - tStart;
        }
        else // 分组计算
        {
            if (maxConcurrentAreas == 0) // 自动分组
                maxConcurrentAreas = nMKL * 3;
            const auto &regionIndices = subProbs.subProb2ArrIdx[probType];

            for (Idx segmentBegin = 0; segmentBegin < regionIndices.size();)
            {
                const Idx segmentSize = std::min(maxConcurrentAreas,
                                                 static_cast<Idx>(regionIndices.size()) - segmentBegin);
                const Idx segmentEnd = segmentBegin + segmentSize;
                const std::vector<Idx> regionSegment(regionIndices.begin() + segmentBegin,
                                                     regionIndices.begin() + segmentEnd);
                double tStart = omp_get_wtime();
                BTQ tempRes = computeBTQ(Q, regionSegment, ifOMP);
                double tEnd = omp_get_wtime();
                runTimeInfo.distributeRHS[probType] += tEnd - tStart;
                tStart = omp_get_wtime();
                if (tempRes.BTQ.cols() > 0)
                    tempRes.BTQ = subProbs.slovers[probType]->solve(tempRes.BTQ);
                tEnd = omp_get_wtime();
                runTimeInfo.FQ[probType] += tEnd - tStart;
                tStart = omp_get_wtime();
                res += computeBABTQ(tempRes, regionSegment, ifOMP);
                tEnd = omp_get_wtime();
                runTimeInfo.collectRHS[probType] += tEnd - tStart;
                segmentBegin = segmentEnd;
            }
        }
        mkl_set_num_threads_local(old_nt);
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::FQ(const RowSparseMat &Q, RunTimeInfo &runTimeInfo, Idx maxConcurrentAreas, bool ifOMP) const
    {
        Idx nSegs = subProbs.nSubProbs.size();
        std::vector<Mat_t<myFEM::Complex>> contributions(nSegs);
        // 开启嵌套并行
        const int previousLevels = omp_get_max_active_levels();
        // const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
        const int threadBudget = omp_get_max_threads();
        for (Idx batchBegin = 0; batchBegin < nSegs;)
        {
            Idx batchEnd = batchBegin;
            int batchThreadCount = 0;
            while (batchEnd < nSegs)
            {
                const int localThreads = subProbs.nloMklThreads[batchEnd];
                if (localThreads <= 0 || localThreads > threadBudget)
                    throw std::logic_error("Invalid local thread count for FQ segment");
                if (batchEnd > batchBegin && batchThreadCount + localThreads > threadBudget)
                    break;
                batchThreadCount += localThreads;
                ++batchEnd;
            }

            const int batchSize = static_cast<int>(batchEnd - batchBegin);
#pragma omp parallel for num_threads(batchSize) schedule(static)
            for (Idx i = batchBegin; i < batchEnd; ++i)
            {
                const int previousOmpThreads = omp_get_max_threads();
                const int localThreads = subProbs.nloMklThreads[i];
                omp_set_num_threads(localThreads);
                contributions[i] = loFQ(Q, runTimeInfo, i, ifOMP, localThreads, maxConcurrentAreas);
                omp_set_num_threads(previousOmpThreads);
            }
            batchBegin = batchEnd;
        }
        omp_set_max_active_levels(previousLevels);
        // mkl_set_dynamic(previousMklDynamic);
        // 合并各个子区域的贡献
        Mat_t<myFEM::Complex> res = Mat_t<myFEM::Complex>::Zero(contributions[0].rows(), contributions[0].cols());
#pragma omp parallel for reduction(matComplexAdd : res)
        for (Idx i = 0; i < nSegs; ++i)
        {
            res += contributions[i];
        }
        return res;
    }
    Mat_t<myFEM::Complex> DDMInfo::computeQHFQ(RunTimeInfo &runTimeInfo, Idx maxBlkSz, Idx maxConcurrentAreas, bool ifOMP) const
    { // 当maxBlkSz=-1时，不对Q分块直接计算QHFQ，否则对Q分块计算QHFQ
        if (coaSpa.Q_full.cols() == 0)
        {
            std::cout << "DDMInfo::computeQHFQ()：请使用QSaveType::full构造粗空间\n";
            std::exit(EXIT_FAILURE);
        }
        if (maxBlkSz == 0)
            throw std::invalid_argument("Maximum Q block size must be positive");

        const Idx dimQ = coaSpa.Q_full.cols();
        Mat_t<myFEM::Complex> res(dimQ, dimQ);
        // 对Q分块
        if (maxBlkSz > 0)
        {
            double tStart = omp_get_wtime();
            const Idx nBlks = (dimQ + maxBlkSz - 1) / maxBlkSz;
            std::vector<std::pair<Idx, Idx>> blks(nBlks);
            Idx pos = 0;
            Idx remaining = dimQ;
            for (Idx i = 0; i < nBlks; ++i)
            {
                const Idx blockSize = std::min(maxBlkSz, remaining);
                blks[i] = {pos, blockSize};
                pos += blks[i].second;
                remaining -= blockSize;
            }
            const myFEM::SparseMat_t<myFEM::Complex> &Q = coaSpa.Q_full;
            double tEnd = omp_get_wtime();
            runTimeInfo.genBlkQ += tEnd - tStart;
            if (ifOMP)
            {
                for (Idx col = 0; col < nBlks; ++col)
                {
                    tStart = omp_get_wtime();
                    RowSparseMat blkQ = Q.block(0, blks[col].first, Q.rows(), blks[col].second);
                    tEnd = omp_get_wtime();
                    runTimeInfo.genBlkQ += tEnd - tStart;
                    tStart = omp_get_wtime();
                    Mat_t<myFEM::Complex> blkFQ = FQ(blkQ, runTimeInfo, maxConcurrentAreas, ifOMP);
                    tEnd = omp_get_wtime();
                    runTimeInfo.totalFQ += tEnd - tStart;
                    tStart = omp_get_wtime();
#pragma omp parallel for
                    for (int row = 0; row < nBlks; ++row)
                    {
                        res.block(blks[row].first, blks[col].first,
                                  blks[row].second, blks[col].second) =
                            Q.block(0, blks[row].first, Q.rows(), blks[row].second).adjoint() * blkFQ;
                    }
                    tEnd = omp_get_wtime();
                    runTimeInfo.QHF += tEnd - tStart;
                }
            }
            else
            {
                RowSparseMat Qadj = Q.adjoint();
                for (Idx col = 0; col < nBlks; ++col)
                {
                    RowSparseMat blkQ = Q.block(0, blks[col].first, Q.rows(), blks[col].second);
                    tStart = omp_get_wtime();
                    Mat_t<myFEM::Complex> blkFQ = FQ(blkQ, runTimeInfo, maxConcurrentAreas, true);
                    tEnd = omp_get_wtime();
                    runTimeInfo.totalFQ += tEnd - tStart;
                    res.middleCols(blks[col].first, blks[col].second) = Qadj * blkFQ;
                }
            }
        }
        else // 不对Q分块，直接计算
        {
            RowSparseMat Q = coaSpa.Q_full;
            double tStart = omp_get_wtime();
            Mat_t<myFEM::Complex> FQres = FQ(Q, runTimeInfo, maxConcurrentAreas, ifOMP);
            double tEnd = omp_get_wtime();
            runTimeInfo.totalFQ += tEnd - tStart;
            res = Q.adjoint() * FQres;
        }
        return res;
    }
    void DDMInfo::initCoarseSpace(std::pair<double, double> ranBAWVec, int nBAWVec,
                                  std::vector<double> vSAW, QSaveType qSaveType, double tol)
    {
        // ranBAWVec:体波波失的角度范围，nBAWVec体波组数
        // vSAW:表面波波速(带量纲), tol:粗空间特征值的阈值
        // 按子区域并行地构造粗空间
        myFEM::Complex J(0, 1); // 虚数单位
        if (subRegs.size() == 0)
        {
            std::cout << "DDMInfo::initCoarseSpace()：请先用deviceAyyar变量初始化DDMInfo变量\n";
            std::exit(EXIT_FAILURE);
        }
        std::cout << "开始构造粗空间...\n";
        std::vector<Mat_t<myFEM::Complex>> Q(subRegs.size() - 1);
        std::vector<Idx> loDim(subRegs.size() - 1, 0);
        std::vector<std::pair<Idx, Idx>> gloLagIdx(subRegs.size() - 1);
        for (int i = 0; i < vSAW.size(); ++i)
            vSAW[i] /= dimScales.v0; // 表面波波速无量纲化
#pragma omp parallel
        {
            // int old_nt = mkl_set_num_threads_local(1);
            Idx nJobs = Q.size();
            int ompSize = omp_get_num_threads();
            int rank = omp_get_thread_num();
            auto jobList = myOMP::distributeTasks(ompSize, nJobs);
            int tStart = jobList[rank].first;
            int tEnd = tStart + jobList[rank].second;
            for (int m = tStart; m < tEnd; ++m)
            {
                Idx probType = subRegs[m].probType;
                // 计算BAW和SAW波的解析解
                int nloBasis = 0; // 粗基数量，即Q当前快的最大列数
                std::vector<SAW2_5D::BAWSol> bawSol;
                std::vector<SAW2_5D::SAWSol> sawSol;
                if (nBAWVec > 0) // 计算体波解
                {
                    auto vecBAW = SAW2_5D::genWaveVecs_xz(ranBAWVec, nBAWVec);
                    for (int i = 0; i < vecBAW.size(); ++i)
                    {
                        bawSol.emplace_back(SAW2_5D::solveBAW(subProbs.para_p[probType], vecBAW[i]));
                        nloBasis += bawSol.back().eigVal.size() * bawSol.back().eigVec.rows();
                    }
                }
                if (vSAW.size() > 0) // 计算表面波解
                {
                    for (int i = 0; i < vSAW.size(); ++i)
                        sawSol.emplace_back(SAW2_5D::solveSAW(subProbs.para_p[probType], vSAW[i]));
                    for (auto &sol : sawSol)
                    {
                        std::vector<Idx> keepIdx;
                        for (Idx j = 0; j < sol.lambda.size(); ++j)
                        {
                            if (sol.lambda(j).imag() < 0.0)
                                keepIdx.emplace_back(j);
                        }
                        // 仅保留负虚部的特征值
                        decltype(sol.lambda) lambda(keepIdx.size());
                        decltype(sol.eigVec) eigVec(sol.eigVec.rows(), keepIdx.size());
                        for (Idx j = 0; j < keepIdx.size(); ++j)
                        {
                            lambda(j) = sol.lambda(keepIdx[j]);
                            eigVec.col(j) = sol.eigVec.col(keepIdx[j]);
                        }
                        sol.lambda = std::move(lambda);
                        sol.eigVec = std::move(eigVec);
                        nloBasis += sol.lambda.size() * sol.eigVec.rows();
                    }
                }
                // 找出当前子区域所有bSign为1的Lagrange乘子对应的自由度索引，并按自由度类型分类
                std::vector<std::vector<Idx>> lagLoIdx(4);                  // bSign为1的Lagrange乘子局部索引；0:x, 1:y, 2:z, 3:phi
                std::vector<std::vector<double>> pt_x(4), pt_y(4), pt_z(4); // bSign为1的Lagrange乘子对应的自由度坐标
                Idx nPosLag = 0;                                            // bSign为1的Lagrange乘子数量
                Idx loStart = -1, loEnd = -1;
                for (Idx i = 0; i < subRegs[m].bLag2Nodes.size(); ++i)
                {
                    if (subRegs[m].bSign[i] == 1)
                    {
                        if (loStart == -1 || loStart > subRegs[m].bGlo2Lo[i])
                            loStart = subRegs[m].bGlo2Lo[i];
                        if (loEnd == -1 || loEnd < subRegs[m].bGlo2Lo[i])
                            loEnd = subRegs[m].bGlo2Lo[i];
                        Idx nodeIdx = subRegs[m].bLag2Nodes[i];
                        Idx lagType = subRegs[m].lagType[i];
                        lagLoIdx[lagType].emplace_back(nPosLag);
                        pt_x[lagType].emplace_back(subRegs[m].dofIdx(nodeIdx, 0));
                        pt_y[lagType].emplace_back(subRegs[m].dofIdx(nodeIdx, 1));
                        pt_z[lagType].emplace_back(subRegs[m].dofIdx(nodeIdx, 2));
                        nPosLag++;
                    }
                }
                gloLagIdx[m] = {loStart, loEnd};
                // 构造Q的该子区域矩阵块
                Q[m].resize(nPosLag, nloBasis); // 只有bSign为1的Lagrange乘子自由度的值非0
                Q[m].setZero();
                Idx posCol = 0; // 当前粗空间矩阵块的列索引
                // 先填入SAW对应部分
                double top_z = subRegs[m].dom_p(2, 1); // 基底上表面z坐标
                for (auto sawBasis : sawSol)
                {
                    for (int i = 0; i < sawBasis.lambda.size(); ++i)
                    {
                        double beta = w / sawBasis.vSAW; // SAW波数
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[0].size(); ++j) // x方向位移
                        {
                            Idx idxTemp_u = lagLoIdx[0][j];
                            myFEM::Complex val = sawBasis.eigVec(0, i) *
                                                 std::exp(-J * beta * (pt_x[0][j] - sawBasis.lambda(i) * (pt_z[0][j] - top_z)));
                            Q[m](idxTemp_u, posCol) = val;
                        }
                        posCol++;
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[1].size(); ++j) // y方向位移
                        {
                            Idx idxTemp_v = lagLoIdx[1][j];
                            myFEM::Complex val = sawBasis.eigVec(1, i) *
                                                 std::exp(-J * beta * (pt_x[1][j] - sawBasis.lambda(i) * (pt_z[1][j] - top_z)));
                            Q[m](idxTemp_v, posCol) = val;
                        }
                        posCol++;
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[2].size(); ++j) // z方向位移
                        {
                            Idx idxTemp_w = lagLoIdx[2][j];
                            myFEM::Complex val = sawBasis.eigVec(2, i) *
                                                 std::exp(-J * beta * (pt_x[2][j] - sawBasis.lambda(i) * (pt_z[2][j] - top_z)));
                            Q[m](idxTemp_w, posCol) = val;
                        }
                        posCol++;
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[3].size(); ++j) // phi
                        {
                            Idx idxTemp_phi = lagLoIdx[3][j];
                            myFEM::Complex val = sawBasis.eigVec(3, i) *
                                                 std::exp(-J * beta * (pt_x[3][j] - sawBasis.lambda(i) * (pt_z[3][j] - top_z)));
                            Q[m](idxTemp_phi, posCol) = val;
                        }
                        posCol++;
                    }
                }
                // 再填入BAW对应部分
                for (auto bawBasis : bawSol)
                {
                    for (int i = 0; i < bawBasis.eigVal.size(); ++i)
                    {
                        auto vBAW = bawBasis.eigVal[i];
                        Eigen::Vector3cd kBAW = w / vBAW * bawBasis.n; // 波失
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[0].size(); ++j) // x方向位移
                        {
                            Idx idxTemp_u = lagLoIdx[0][j];
                            myFEM::Complex val = bawBasis.eigVec(0, i) *
                                                 std::exp(-J * (kBAW(0) * pt_x[0][j] +
                                                                kBAW(1) * pt_y[0][j] +
                                                                kBAW(2) * pt_z[0][j]));
                            Q[m](idxTemp_u, posCol) = val;
                        }
                        posCol++;
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[1].size(); ++j) // y方向位移
                        {
                            Idx idxTemp_v = lagLoIdx[1][j];
                            myFEM::Complex val = bawBasis.eigVec(1, i) *
                                                 std::exp(-J * (kBAW(0) * pt_x[1][j] +
                                                                kBAW(1) * pt_y[1][j] +
                                                                kBAW(2) * pt_z[1][j]));
                            Q[m](idxTemp_v, posCol) = val;
                        }
                        posCol++;
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[2].size(); ++j) // z方向位移
                        {
                            Idx idxTemp_w = lagLoIdx[2][j];
                            myFEM::Complex val = bawBasis.eigVec(2, i) *
                                                 std::exp(-J * (kBAW(0) * pt_x[2][j] +
                                                                kBAW(1) * pt_y[2][j] +
                                                                kBAW(2) * pt_z[2][j]));
                            Q[m](idxTemp_w, posCol) = val;
                        }
                        posCol++;
#pragma omp simd
                        for (Idx j = 0; j < lagLoIdx[3].size(); ++j) // phi
                        {
                            Idx idxTemp_phi = lagLoIdx[3][j];
                            myFEM::Complex val = bawBasis.eigVec(3, i) *
                                                 std::exp(-J * (kBAW(0) * pt_x[3][j] +
                                                                kBAW(1) * pt_y[3][j] +
                                                                kBAW(2) * pt_z[3][j]));
                            Q[m](idxTemp_phi, posCol) = val;
                        }
                        posCol++;
                    }
                }
                // 通过对Q[m]进行SVD分解，保留奇异值大于tol的左奇异向量作为粗空间基
                if (Q[m].cols() > 0)
                {
                    Eigen::JacobiSVD<Mat_t<myFEM::Complex>> svd(Q[m], Eigen::ComputeThinU);
                    const auto &singularValues = svd.singularValues();
                    Idx nKeep = 0;
                    while (nKeep < singularValues.size() && singularValues(nKeep) > tol)
                        ++nKeep;
                    Q[m] = svd.matrixU().leftCols(nKeep);
                    loDim[m] = nKeep;
                }
            }
            // mkl_set_num_threads_local(old_nt);
        }
        switch (qSaveType)
        {
        case blocks:
        {
            coaSpa.Q_blk = std::move(Q);
            coaSpa.loDim = std::move(loDim);
            coaSpa.gloLagIdx = std::move(gloLagIdx);
            coaSpa.gloBasIdx.resize(coaSpa.Q_blk.size());
            if (!coaSpa.Q_blk.empty())
            {
                coaSpa.gloBasIdx[0] = {0, coaSpa.loDim[0] - 1};
                for (Idx m = 1; m < coaSpa.Q_blk.size(); ++m)
                {
                    coaSpa.gloBasIdx[m] = {coaSpa.gloBasIdx[m - 1].second + 1,
                                           coaSpa.gloBasIdx[m - 1].second + coaSpa.loDim[m]};
                }
            }
            break;
        }
        case full:
        {
            coaSpa.Q_full = myEigen::blkdiag<myFEM::Complex, Eigen::ColMajor>(Q);
            break;
        }
        }
    }
    Mat_t<myFEM::Complex> DDMInfo::Py(const Mat_t<myFEM::Complex> &V, QSaveType qSaveType, bool ifOMP) const
    { // 本函数计算Py = (I-Q(Q^HFQ)^{-1}Q^HF)y
        if (V.rows() != nLags)
            throw std::invalid_argument("Py input row count must equal the number of Lagrange multipliers");

        // 先计算 F*y，再投影到粗空间并求解粗问题。
        Mat_t<myFEM::Complex> res = Fy(V, ifOMP);
        res = QHmulV(res, qSaveType, ifOMP);
        res = solveCoaProb(res);
        res = QmulV(res, qSaveType, ifOMP);
        return V - res;
    }
    Vec_t<myFEM::Complex> DDMInfo::initLagsInCoaSpa(const Vec_t<myFEM::Complex> &lagRHS, QSaveType qSaveType, bool ifOMP) const
    {
        // 计算lagrange乘子在粗空间中的初值，调用本函数要保证已完成粗问题构造
        // lagRHS为lagrange乘子线性方程组的右端项
        Vec_t<myFEM::Complex> res = QHmulV(lagRHS, qSaveType, ifOMP);
        res = solveCoaProb(res);
        return QmulV(res, qSaveType, ifOMP);
    }
    DDMInfo::IterationSolverResults DDMInfo::GCR_TwoLevel_DirichletPrecond(int maxit, double epcl, QSaveType qSaveType, bool ifOMP) const
    {
        using Vec = Vec_t<myFEM::Complex>;
        std::vector<double> res;
        int iter = 0;
        Vec d = computeLagRHS(ifOMP);
        double dNrom = d.norm();
        Vec lamd = initLagsInCoaSpa(d, qSaveType, ifOMP); // 粗空间解
        Vec r = Fy(lamd, ifOMP);
        r = d - r;
        res.emplace_back(r.norm() / dNrom);
        if (res.back() < epcl)
        {
            std::cout << "Converged in " << iter << " iterations, residual =" << res[iter] << ".\n";
            return IterationSolverResults(lamd, iter, res, true);
        }
        std::vector<Vec> p, Fp, MFp;
        std::vector<double> MFpMFp;
        p.reserve(maxit), Fp.reserve(maxit), MFp.reserve(maxit), MFpMFp.reserve(maxit);

        Vec z = My(r, ifOMP);
        p.emplace_back(Py(z, qSaveType, ifOMP));
        Fp.emplace_back(Fy(p.back(), ifOMP));
        MFp.emplace_back(My(Fp.back(), ifOMP));
        MFpMFp.emplace_back(MFp.back().squaredNorm());
        for (; iter < maxit;)
        {
            ++iter;
            myFEM::Complex alpha = MFp.back().dot(z) / MFpMFp.back();
            lamd += alpha * p.back();
            r -= alpha * Fp.back();
            res.emplace_back(r.norm() / dNrom);
            std::cout << "iteration " << iter << ", " << "res = " << res.back() << "\n";
            if (res.back() < epcl)
            {
                std::cout << "Converged in " << iter << " iterations, residual =" << res[iter] << ".\n";
                return IterationSolverResults(lamd, iter, res, true);
            }
            z -= alpha * MFp.back();
            Vec y = Py(z, qSaveType, ifOMP), fy = Fy(y, ifOMP), MFy = My(fy, ifOMP);
            Vec beta(iter);
#pragma omp parallel for if (ifOMP)
            for (int j = 0; j < iter; ++j)
            {
                beta(j) = -MFp[j].dot(MFy) / MFpMFp[j];
            }
            for (int j = 0; j < iter; ++j)
            {
                y += p[j] * beta(j);
                fy += Fp[j] * beta(j);
                MFy += MFp[j] * beta(j);
            }
            p.emplace_back(y);
            Fp.emplace_back(fy);
            MFp.emplace_back(MFy);
            MFpMFp.emplace_back(MFy.squaredNorm());
        }
        return IterationSolverResults(lamd, iter, res, false);
    }
    void DDMInfo::computeDispAndElePotential(Vec_t<myFEM::Complex> lambda, bool ifNd, bool ifOMP)
    {
        // 本函数通过输入的lambda计算subRegs[i].uh (i=0,...,subRegs[i].size()-1)
        // 用omp task实现按子问题类型并行，对typeProb子问题线程，首先若subProbs.loSol[typeProb].size()>0则将
        // subRegs[i].uh=subProbs.loSol[typeProb] (i in subProbs.subProb2ArrIdx[typeProb]), 该步骤按omp for并行，或者串行，按你认为最高效的方式
        // 若subProbs.loSol[typeProb].size()=0，则令subRegs[i].uh为长度等于subRegs[i].dofIdx.rows()的零向量
        // 然后对于typeProb类对应的子区域，首先执行distributeRHS(lambda, typeProb, ifOMP)，然后对输出结果执行subProbs.slovers[typeProb]->solve
        // 最后会得到一个稠密矩阵，然后将这个稠密矩阵列按subProbs.subProb2ArrIdx[typeProb]中子区域编号并行地加到对应子区域的subRegs[i].uh
        // 编写这个程序时你可以参考loFy和Fy函数的结构
        if (lambda.size() != nLags)
            throw std::invalid_argument("Lambda size must equal the number of Lagrange multipliers");

        const Idx nSegs = subProbs.nSubProbs.size();
#pragma omp parallel for if (ifOMP)
        for (Idx i = 0; i < subRegs.size(); ++i)
        {
            const Idx typeProb = subRegs[i].probType;
            if (subProbs.loSol[typeProb].size() > 0)
                subRegs[i].uh = subProbs.loSol[typeProb];
            else
                subRegs[i].uh = Vec_t<myFEM::Complex>::Zero(subRegs[i].dofIdx.rows());
        }

        const int previousLevels = omp_get_max_active_levels();
        const int previousMklDynamic = mkl_get_dynamic();
        omp_set_max_active_levels(2);
        mkl_set_dynamic(0);
#pragma omp parallel
        {
#pragma omp single
            {
                for (Idx typeProb = 0; typeProb < nSegs; ++typeProb)
                {
#pragma omp task firstprivate(typeProb)
                    {
                        const int previousOmpThreads = omp_get_max_threads();
                        omp_set_num_threads(subProbs.nloMklThreads[typeProb]);
                        Mat_t<myFEM::Complex> rhs = distributeRHS(lambda, typeProb, ifOMP);
                        Mat_t<myFEM::Complex> localSol = subProbs.slovers[typeProb]->solve(rhs);
                        const auto &arrIdx = subProbs.subProb2ArrIdx[typeProb];
#pragma omp parallel for if (ifOMP)
                        for (Idx j = 0; j < arrIdx.size(); ++j)
                            subRegs[arrIdx[j]].uh -= localSol.col(j);
                        if (ifNd)
                        {
#pragma omp parallel for if (ifOMP)
                            for (Idx j = 0; j < arrIdx.size(); ++j)
                            {
                                Idx domIdx = arrIdx[j];
                                Mat_t<double> *ptrDofIdx = &subRegs[domIdx].dofIdx;
                                Vec_t<myFEM::Complex> *ptrUh = &subRegs[domIdx].uh;

                                ptrDofIdx->leftCols(3) *= dimScales.L0;
#pragma omp simd
                                for (Idx nDof = 0; nDof < ptrDofIdx->rows(); ++nDof)
                                {
                                    if (ptrDofIdx->coeff(nDof, 3) >= 0 && ptrDofIdx->coeff(nDof, 3) <= 2)
                                        (*ptrUh)(nDof) *= dimScales.U0;
                                    else if (ptrDofIdx->coeff(nDof, 3) == 3)
                                    {
                                        (*ptrUh)(nDof) *= dimScales.Phi0;
                                    }
                                }
                            }
                        }
                        omp_set_num_threads(previousOmpThreads);
                    }
                }
#pragma omp taskwait
            }
        }
        omp_set_max_active_levels(previousLevels);
        mkl_set_dynamic(previousMklDynamic);
    }
    /*模板*/
    template Mat_t<myFEM::Complex> DDMInfo::distributeRHS(const RowSparseMat &V, Idx probType, bool ifOMP) const;
    template Mat_t<myFEM::Complex> DDMInfo::distributeRHS(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP) const;
    template Mat_t<myFEM::Complex> DDMInfo::loFy(const RowSparseMat &V, Idx probType, bool ifOMP, int nMKL) const;
    template Mat_t<myFEM::Complex> DDMInfo::loFy(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP, int nMKL) const;
    template Mat_t<myFEM::Complex> DDMInfo::Fy_Serial(const RowSparseMat &V) const;
    template Mat_t<myFEM::Complex> DDMInfo::Fy_Serial(const Mat_t<myFEM::Complex> &V) const;
    template Mat_t<myFEM::Complex> DDMInfo::Fy(const RowSparseMat &V, bool ifOMP) const;
    template Mat_t<myFEM::Complex> DDMInfo::Fy(const Mat_t<myFEM::Complex> &V, bool ifOMP) const;
};
