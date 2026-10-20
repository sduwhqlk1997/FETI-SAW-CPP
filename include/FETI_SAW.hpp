#pragma once
#include <Eigen/Dense>
#include <vector>
#include <SAW.hpp>
#include <mkl.h>
#include <Eigen/PardisoSupport>
namespace FETI
{
    using RowSparseMat = Eigen::SparseMatrix<myFEM::Complex, Eigen::RowMajor>;
    using Idx = Eigen::Index;
    template <typename Scalar>
    using Vec_t = Eigen::Vector<Scalar, Eigen::Dynamic>;
    template <typename Scalar>
    using Mat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Solver = Eigen::PardisoLDLT<myFEM::SparseMat_t<myFEM::Complex>, Eigen::Upper | Eigen::Symmetric>;
    class SubRegion // 子区域
    {
        // 成员变量
    public:
        Idx probType; // 子区域对应的子问题类型编号
        // 子区域几何信息与自由度信息
        Mat_t<double> dom_p, dom_e;
        Mat_t<double> dofIdx;
        // Lagrange乘子信息
        std::vector<Idx> bGlo2Lo;    // 全局lagrange乘子的编号映射到该子区域局部Lagrange乘子编号，索引为局部编号，元素为全局编号
        std::vector<Idx> bLag2Nodes; // 局部lagrange乘子与局部网格点自由度对应关系，行号为lagrange乘子的局部编号，元素为自由度编号(与Dofindex的行对应)
        std::vector<int> bSign;      // 局部Lagrange乘子在该区域对应的符号，为1和-1组成的矩阵
        std::vector<int> lagType;    // 局部Lagrange对应的自由度类型，行号为Lagrange乘子的局部编号，元素为对应Lagrange乘子的类型
        // 界面自由度信息
        std::vector<Idx> intDofIdx; // 界面自由度局部编号，长度与dofIdx.rows()相同，内部自由度值为-1，界面自由度对应的值为界面自由度的局部编号，即仅对界面自由度进行编号得到的编号
        Idx numIntDof = 0;          // 界面自由度数量
        // 内部自由度信息
        std::vector<Idx> innDofIdx; // 内部自由度局部编号，长度与dofIdx.rows()相同，内部自由度对应的值为内部自由度的局部编号，即仅对内部自由度进行编号得到的编号，界面自由度值为-1
        Idx numInnDof = 0;          // 内部自由度数量
        // 数值解信息
        Vec_t<myFEM::Complex>
            uh;

    private:
        // 构造函数
    public:
        SubRegion() {};
        SubRegion(Mat_t<double> dom_p, Mat_t<double> dom_e, const Mat_t<double> &dofIdx)
            : dom_p(dom_p), dofIdx(dofIdx)
        {
            if (dom_e.rows() > 0)
                this->dom_e = dom_e;
        }
        // 生成界面和内部自由度信息
        void genIntInnDofInfo(); // 生成界面和内部自由度信息
        // B矩阵参与的运算
        Mat_t<myFEM::Complex> BmulV(const Mat_t<myFEM::Complex> &V, Idx numLag) const; // 计算B*Vec
        Mat_t<myFEM::Complex> BmulV(const RowSparseMat &V, Idx numLag) const;
        Mat_t<myFEM::Complex> BTmulV(const Mat_t<myFEM::Complex> &V) const; // 计算B^T*Vec
        Mat_t<myFEM::Complex> BTmulV(const RowSparseMat &V) const;
        myFEM::SparseMat_t<myFEM::Complex> BTmulV_sp(const RowSparseMat &V) const; // 计算B^T*Vec并以稀疏矩阵返回
        // 用于Dirichlet预条件子乘矩阵的运算
        Mat_t<myFEM::Complex> BmulV_int(const Mat_t<myFEM::Complex> &V, Idx numLag) const; // 计算B*Vec，但输出的矩阵的列对应界面自由度
        Mat_t<myFEM::Complex> BTmulV_int(const Mat_t<myFEM::Complex> &V) const;            // 计算B^T*Vec，但输入的矩阵的行对应界面自由度
        template <typename MatType>
        Mat_t<myFEM::Complex> distributeRHS(const MatType &V, std::vector<Idx> subProbIdxs, Idx probType) const; // 收集所有type类子区域的右端项
        // 其他工具函数
    public:
        Idx getLoNumLag() const { return bGlo2Lo.size(); }
    };
    // 工具函数声明
    inline Idx idx1D(Idx i, Idx j, Idx k, Idx nx, Idx ny)
    {
        return i + j * nx + k * nx * ny;
    }
    inline SubRegion &getSubReg(Idx i, Idx j, Idx k, Idx nx, Idx ny, Idx nz, std::vector<SubRegion> &subRegs)
    {
        return subRegs[idx1D(i, j, k, nx, ny)];
    }

    // 由SAW器件序列初始化子区域对象
    std::vector<SubRegion> initSubRegsFromDevArr(const SAW2_5D::deviceArray &devArr); // 按x->y->z方向排序为一维SubRegion向量
    // 生成Lagrange乘子信息
    std::vector<std::unordered_map<Vec_t<double>, Idx,
                                   myFEM::PointNdHash, myFEM::PointNdEqual>>
    genSubRegBdIdx(SubRegion &subReg); // 生成子区域左右、后前、下上面的坐标-索引Hash表
    std::pair<std::vector<Idx>, std::vector<Idx>> findCommDofOnSingleFace(std::unordered_map<Vec_t<double>, Idx,
                                                                                             myFEM::PointNdHash, myFEM::PointNdEqual>
                                                                              face1,
                                                                          std::unordered_map<Vec_t<double>, Idx,
                                                                                             myFEM::PointNdHash, myFEM::PointNdEqual>
                                                                              face2); // 串行地找到交界面上公共点的一一对应关系
    Idx genLagInfo(std::vector<SubRegion> &subRegs, Idx nx, Idx ny, Idx nz);          // 返回全局Lagrange乘子总数
    // 区域分解算法所需数据整合
    class DDMInfo
    {
        /*局部数据结构定义*/
    private:
        struct SchurComp // S = Kbb-Kbi*Kii^{-1}*Kib
        {
            std::unique_ptr<Solver> invKii =
                std::make_unique<Solver>();                   // 交界面自由度对应矩阵块的LDL分解
            myFEM::SparseMat_t<myFEM::Complex> Kbb, Kbi, Kib; // 矩阵块
        };

        struct SubProbInfo // 存储所有子问题的信息
        {
            // 子问题信息
            std::vector<SAW2_5D::material> para_p;        // 子问题物理区域的基底材料参数，大小为子问题种类数
                                                          // 若为PML区域则为相连的物理区域的参数，
                                                          // 一个子问题的物理区域只能包含一种基底材料。
            std::vector<int> startIdxSubProbType;         // 第i个元素为第i类子问题在subProb2ArrIdx、nSubProbs和nloMklThreads的起点索引，大小为子问题种类数
            std::vector<int> subProbType;                 // 各子问题的类型
            std::vector<std::vector<Idx>> subProb2ArrIdx; // 每类子问题对应的子区域索引，每行表示一个子问题的一个片段（按子问题序号依次排列，下同），其中元素为子区域编号
            std::vector<Idx> nSubProbs;                   // 各子问题数量

            std::vector<std::unique_ptr<Solver>> slovers; // 各类子问题（片段）的LDLT分解
            std::vector<Vec_t<myFEM::Complex>> RHS;       // 每类子问题只保存一份，大小为子问题种类数
            std::vector<Vec_t<myFEM::Complex>> loSol;     // 各类子问题的K^{-1}*RHS，大小为子问题种类数
            // 并行设置
            std::vector<int> nloMklThreads; // 每个子问题（片段）计算时可使用的mkl线程数
            // 预条件子 (todo)
        };

        struct CoarseSpace
        {
            std::vector<SAW2_5D::Mat_t<myFEM::Complex>> Q_blk; // 粗空间，按子区域分块
            myFEM::SparseMat_t<myFEM::Complex> Q_full;         // 粗空间，稀疏存储
            std::vector<std::pair<Idx, Idx>> gloLagIdx;        // 局部粗空间对应的全局Lagrange乘子索引，即gloLagIdx[i].first--gloLagIdx[i].second为Q[i]对应的全局Lagrange乘子索引范围
            std::vector<std::pair<Idx, Idx>> gloBasIdx;        // 局部粗空间对应的全局基函数索引，即gloBasIdx[i].first--gloBasIdx[i].second为Q[i]对应的全局基函数索引范围
            std::vector<Idx> loDim;                            // 局部粗空间维数
            //(Q^HFQ)^-1的LU分解
            Mat_t<myFEM::Complex> luFactor; // zgetrf 原地写入的通用 LU 分解数据
            std::vector<MKL_INT> luPivots;  // zgetrf 的主元信息，供 zgetrs 复用
        };
        struct BTQ // B^T*Q
        {
            myFEM::Mat_t<myFEM::Complex> BTQ;          // B^T*Q但仅保留非0列
            std::vector<std::vector<Idx>> loIdx;       // 向量的每个元素对应一个子区域，在内部的vector中元素与相应的BTQ的列对应，表示该列对应的局部编号(也就是矩阵Q的列索引)
            std::vector<std::pair<Idx, Idx>> segRange; // BTQ的列对应的全局Lagrange乘子索引范围，即segRange[i].first--segRange[i].second为BTQ[i]对应的全局Lagrange乘子索引范围
            Idx dimQ = 0;                              // Q的总维数
            // FQ计算思路:
            // Q稀疏存储，定义distributeRHS_sp函数，通过调用SubRegs.BTmulV_sp将同类型子问题的B^TQ存在一个向量std::vector<myFEM::SparseMat_t<myFEM::Complex>> tempBTQ中
            // 然后遍历这个向量取出所有非零列，并存在BTQ结构体中的BTQ中，并同步形成loIdx和segRanges，这一步实际上可以先遍历tempBTQ并记录非零行位置，并形成loIdx和segRange，
            // 然后用omp for并行地将非零列填入BTQ，可以创建一个gloIdx的辅助vector用来记录非0列与BTQ列索引对应关系，方便并行，其实也没必要并行
            // 计算subProbs[probType].solvers->compute(BTQ)并覆盖BTQ
            // 按segRange记录的子区域分段信息并行地对BTQ按列每段执行SubReg.BmulV(BTQ.middleCol(.,.))
            // 将上一步计算结果每列按loIdx记录的局部编号累加到FQ稠密矩阵中得到FQ
        };

    public:
        enum QSaveType
        {
            blocks, // 保存所有子区域的Q
            full    // 仅保存非零列的Q
        };
        /*成员变量*/
    public:
        double w = 0;                      // 工作频率
        std::vector<SubRegion> subRegs;    // Lagrange乘子、自由度索引
        SubProbInfo subProbs;              // 子区域段，每个元素为一个子区域段，其中存储这该段子区域的子问题和预条件子信息
        SAW2_5D::DimScales dimScales;      // 子问题的无量纲化特征量
        Idx nLags = 0;                     // Lagrange乘子数
        std::vector<SchurComp> dirPrecond; // Dirichlet预条件子
        CoarseSpace coaSpa;                // 粗空间

        // std::vector<int> nloMklThreads; // 每类子区域计算时可使用的mkl线程数
    private:
        /*构造函数*/
    public:
        DDMInfo() {};
        DDMInfo(const SAW2_5D::deviceArray &devArr, Idx nSeg = 1, int segStrategy = 0); // 不初始化材料信息
        DDMInfo(const SAW2_5D::deviceArray &devArr,
                std::vector<SAW2_5D::typeMaterial> &subProbMatTypes, Idx nSeg = 1, int segStrategy = 0); // 初始化材料信息
        /*求解子问题*/
    public:
        void computeloSol();
        /*计算Fλ=d中的d*/
    public:
        Vec_t<myFEM::Complex> computeLagRHS(bool ifOMP = false) const; // 计算Fλ=d中的d
        /*构造右端项*/
    public:
        template <typename MatType>
        Mat_t<myFEM::Complex> distributeRHS(const MatType &V, Idx probType, bool ifOMP = false) const;            // 将右端项分发到各个子区域
        Mat_t<myFEM::Complex> collectRHS(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP = false) const; // 收集各个子区域的右端项
        /*Fy*/
    private:
        template <typename MatType>
        Mat_t<myFEM::Complex> loFy(const MatType &V, Idx probType, bool ifOMP = false, int nMKL = 1) const; // 计算类型probType的所有子区域问题，并将结果合并
    public:
        template <typename MatType>
        Mat_t<myFEM::Complex> Fy(const MatType &V, bool ifOMP = false) const; // 计算所有子区域问题并合并
        template <typename MatType>
        Mat_t<myFEM::Complex> Fy_Serial(const MatType &V) const; // 计算所有子区域问题并合并(串行版本)
        /*预条件子*/
    private:
        void genAllIntInnDofInfo(bool ifOMP = false); // 生成所有子区域的界面自由度信息
        void computeSchurComplement(SchurComp &result,
                                    SubRegion &subReg,
                                    const myFEM::SparseMat_t<myFEM::Complex> &K,
                                    int tMKL = 1);
        Mat_t<myFEM::Complex> distributeRHS_int(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP = false) const;  // 将右端项分发到各个子区域的界面自由度
        Mat_t<myFEM::Complex> collectRHS_int(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP = false) const;     // 收集各个子区域的界面自由度的右端项
        Mat_t<myFEM::Complex> loMy(const Mat_t<myFEM::Complex> &V, Idx probType, bool ifOMP = false, int nMKL = 1) const; // 对probType类型子区域对应的右端项作用预条件子
    public:
        void initDirPrecond(const SAW2_5D::deviceArray &devArr);                            // 初始化Dirichlet预条件子
        Mat_t<myFEM::Complex> My(const Mat_t<myFEM::Complex> &V, bool ifOMP = false) const; // 对所有子区域对应的右端项作用预条件子
        /*粗空间*/
    public:
        struct RunTimeInfo
        {
            double genBlkQ = 0;                // 对Q分块的总耗时
            double QHF = 0;                    // 左乘Q^H的总耗时
            double totalFQ = 0;                // FQ的总耗时
            std::vector<double> distributeRHS; // 每段子问题分发右端项的总耗时
            std::vector<double> FQ;            // 每段子问题计算FQ的总耗时
            std::vector<double> collectRHS;    // 每段子问题收集右端项的总耗时
        };

    private:
        BTQ computeBTQ(const RowSparseMat &Q, const std::vector<Idx> &regionSegment,
                       bool ifOMP = false) const; // 计算一段区域的B^T*Q
        Mat_t<myFEM::Complex> computeBABTQ(BTQ &ABTQ, const std::vector<Idx> &regionSegment,
                                           bool ifOMP = false) const; // 计算一段区域的Q^HFQ
        Mat_t<myFEM::Complex> loFQ(const RowSparseMat &Q, RunTimeInfo &runTimeInfo, Idx probType, bool ifOMP = false,
                                   int nMKL = 1, Idx maxConcurrentAreas = -1) const;                                                           // 分段计算FQ
        Mat_t<myFEM::Complex> FQ(const RowSparseMat &Q, RunTimeInfo &runTimeInfo, Idx maxConcurrentAreas = -1, bool ifOMP = false) const;      // 计算FQ
        Mat_t<myFEM::Complex> computeQHFQ(RunTimeInfo &runTimeInfo, Idx maxBlkSz = -1, Idx maxConcurrentAreas = -1, bool ifOMP = false) const; // 按矩阵Q分块计算QHFQ，注意此时矩阵Q的生成类型应为QSaveType::full
    public:
        void initCoarseSpace(std::pair<double, double> ranBAWVec, int nBAWVec,
                             std::vector<double> vSAW, QSaveType qSaveType = blocks, double tol = 1e-4);                      // 初始化粗空间
        Mat_t<myFEM::Complex> QmulV(const Mat_t<myFEM::Complex> &V, QSaveType qSaveType = blocks, bool ifOMP = false) const;  // 计算Q*Vec
        Mat_t<myFEM::Complex> QHmulV(const Mat_t<myFEM::Complex> &V, QSaveType qSaveType = blocks, bool ifOMP = false) const; // 计算Q^H*Vec
        void initCoarseProblem(std::pair<double, double> ranBAWVec, int nBAWVec,
                               std::vector<double> vSAW, RunTimeInfo &runTimeInfo, double tol = 1e-4,
                               QSaveType qSaveType = blocks, Idx maxConcurrentAreas = -1, Idx maxBlkSz = -1, bool ifOMP = false); // 初始化粗空间问题
        Mat_t<myFEM::Complex> solveCoaProb(const Mat_t<myFEM::Complex> &b) const;                                                 // 使用粗空间LDL^T分解求解稠密线性方程组
        Mat_t<myFEM::Complex> Py(const Mat_t<myFEM::Complex> &V, QSaveType qSaveType = blocks, bool ifOMP = false) const;         // 计算Py，即向粗空间补集的投影
        Vec_t<myFEM::Complex> initLagsInCoaSpa(const Vec_t<myFEM::Complex> &lagRHS, QSaveType qSaveType = blocks, bool ifOMP = false) const;
        /*迭代法求解器*/
    public:
        struct IterationSolverResults
        {
            Vec_t<myFEM::Complex> lambda; // 数值解
            int iter = 0;                 // 迭代次数
            std::vector<double> res;      // 迭代历史相对残差
            bool ifConv = false;          // 迭代是否收敛
            inline IterationSolverResults(Vec_t<myFEM::Complex> lambda,
                                          int iter,
                                          std::vector<double> res,
                                          bool ifConv) : lambda(lambda),
                                                         iter(iter),
                                                         res(res),
                                                         ifConv(ifConv) {}
        };

        IterationSolverResults GCR_TwoLevel_DirichletPrecond(int maxit = 50, double epcl = 1e-4, QSaveType qSaveType = blocks, bool ifOMP = false) const;
        /*计算位移、电势解*/
        void computeDispAndElePotential(Vec_t<myFEM::Complex> lambda, bool ifNd, bool ifOMP = false);
    };
};