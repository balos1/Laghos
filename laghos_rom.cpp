#include "laghos_rom.hpp"

#include "DEIM.h"
#include "SampleMesh.hpp"


using namespace std;

void ROM_Sampler::SampleSolution(const double t, const double dt, Vector const& S)
{
  SetStateVariables(S);
  SetStateVariableRates(dt);
  
  const int tmp = generator_X->getNumBasisTimeIntervals();
  
  const bool sampleX = generator_X->isNextSample(t);

  if (sampleX)
    {
      if (rank == 0)
	{
	  cout << "X taking sample at t " << t << endl;
	}
      
      generator_X->takeSample(X.GetData(), t, dt);
      generator_X->computeNextSampleTime(X.GetData(), dXdt.GetData(), t);
    }

  const bool sampleV = generator_V->isNextSample(t);

  if (sampleV)
    {
      if (rank == 0)
	{
	  cout << "V taking sample at t " << t << endl;
	}
      
      generator_V->takeSample(V.GetData(), t, dt);
      generator_V->computeNextSampleTime(V.GetData(), dVdt.GetData(), t);
    }

  const bool sampleE = generator_E->isNextSample(t);

  if (sampleE)
    {
      if (rank == 0)
	{
	  cout << "E taking sample at t " << t << endl;
	}
      
      generator_E->takeSample(E.GetData(), t, dt);
      generator_E->computeNextSampleTime(E.GetData(), dEdt.GetData(), t);
    }
}

void BasisGeneratorFinalSummary(CAROM::SVDBasisGenerator* bg)
{
  const int rom_dim = bg->getSpatialBasis()->numColumns();
  cout << "ROM dimension = " << rom_dim << endl;

  const CAROM::Matrix* sing_vals = bg->getSingularValues();
            
  cout << "Singular Values:" << endl;
  for (int sv = 0; sv < sing_vals->numColumns(); ++sv) {
    cout << (*sing_vals)(sv, sv) << endl;
  }
}

void ROM_Sampler::Finalize(const double t, const double dt, Vector const& S)
{
  SetStateVariables(S);
  
  generator_X->takeSample(X.GetData(), t, dt);
  generator_X->endSamples();

  generator_V->takeSample(V.GetData(), t, dt);
  generator_V->endSamples();

  generator_E->takeSample(E.GetData(), t, dt);
  generator_E->endSamples();

  if (rank == 0)
    {
      cout << "X basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_X);

      cout << "V basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_V);

      cout << "E basis summary output" << endl;
      BasisGeneratorFinalSummary(generator_E);
    }

  delete generator_X;
  delete generator_V;
  delete generator_E;
}

CAROM::Matrix* GetFirstColumns(const int N, const CAROM::Matrix* A, const int rowOS, const int numRows)
{
  CAROM::Matrix* S = new CAROM::Matrix(numRows, std::min(N, A->numColumns()), A->distributed());
  for (int i=0; i<S->numRows(); ++i)
    {
      for (int j=0; j<S->numColumns(); ++j)
	(*S)(i,j) = (*A)(rowOS + i, j);
    }

  return S;
}

CAROM::Matrix* ReadBasisROM(const int rank, const std::string filename, const int vectorSize, const int rowOS, int& dim)
{
  CAROM::BasisReader reader(filename);
  const CAROM::Matrix *basis = (CAROM::Matrix*) reader.getSpatialBasis(0.0);
  
  if (dim == -1)
    dim = basis->numColumns();

  // Make a deep copy of basis, which is inefficient but necessary since BasisReader owns the basis data and deletes it when BasisReader goes out of scope.
  // An alternative would be to keep all the BasisReader instances as long as each basis is kept, but that would be inconvenient.
  CAROM::Matrix* basisCopy = GetFirstColumns(dim, basis, rowOS, vectorSize);

  MFEM_VERIFY(basisCopy->numRows() == vectorSize, "");

  if (rank == 0)
    cout << "Read basis " << filename << " of dimension " << basisCopy->numColumns() << endl;
  
  //delete basis;
  return basisCopy;
}

ROM_Basis::ROM_Basis(MPI_Comm comm_, ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, 
		     int & dimX, int & dimV, int & dimE,
		     const bool staticSVD_, const bool hyperreduce_)
  : comm(comm_), tH1size(H1FESpace->GetTrueVSize()), tL2size(L2FESpace->GetTrueVSize()),
    H1size(H1FESpace->GetVSize()), L2size(L2FESpace->GetVSize()),
    gfH1(H1FESpace), gfL2(L2FESpace), 
    rdimx(dimX), rdimv(dimV), rdime(dimE), staticSVD(staticSVD_), hyperreduce(hyperreduce_)
{
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &rank);

  Array<int> osH1(nprocs+1);
  Array<int> nH1(nprocs);
  Array<int> osL2(nprocs+1);
  MPI_Allgather(&tH1size, 1, MPI_INT, osH1.GetData(), 1, MPI_INT, comm);
  MPI_Allgather(&tL2size, 1, MPI_INT, osL2.GetData(), 1, MPI_INT, comm);

  for (int i=nprocs-1; i>=0; --i)
    {
      nH1[i] = osH1[i];
      osH1[i+1] = osH1[i];
      osL2[i+1] = osL2[i];
    }

  osH1[0] = 0;
  osL2[0] = 0;

  osH1.PartialSum();
  osL2.PartialSum();

  rowOffsetH1 = osH1[rank];
  rowOffsetL2 = osL2[rank];

  fH1 = new CAROM::Vector(tH1size, true);
  fL2 = new CAROM::Vector(tL2size, true);

  mfH1.SetSize(tH1size);
  mfL2.SetSize(tL2size);
  
  ReadSolutionBases();

  rX = new CAROM::Vector(rdimx, false);
  rV = new CAROM::Vector(rdimv, false);
  rE = new CAROM::Vector(rdime, false);

  dimX = rdimx;
  dimV = rdimv;
  dimE = rdime;

  if (hyperreduce)
    {
      SetupHyperreduction(H1FESpace, L2FESpace, nH1);
    }
}

void ROM_Basis::SetupHyperreduction(ParFiniteElementSpace *H1FESpace, ParFiniteElementSpace *L2FESpace, Array<int>& nH1)
{
  ParMesh *pmesh = H1FESpace->GetParMesh();
  
  int numSamplesX = rdimx;
  vector<int> sample_dofs_X(numSamplesX);
  vector<int> num_sample_dofs_per_procX(nprocs);
  CAROM::Matrix BsinvX(numSamplesX, rdimx, false);

  int numSamplesV = rdimv;
  vector<int> sample_dofs_V(numSamplesV);
  vector<int> num_sample_dofs_per_procV(nprocs);
  CAROM::Matrix BsinvV(numSamplesV, rdimv, false);

  int numSamplesE = rdime;
  vector<int> sample_dofs_E(numSamplesE);
  vector<int> num_sample_dofs_per_procE(nprocs);
  CAROM::Matrix BsinvE(numSamplesE, rdime, false);
      
  // Perform DEIM or GNAT to find sample DOF's.
  CAROM::GNAT(basisX,
	      rdimx,
	      sample_dofs_X.data(),
	      num_sample_dofs_per_procX.data(),
	      BsinvX,
	      rank,
	      nprocs,
	      numSamplesX);

  CAROM::GNAT(basisV,
	      rdimv,
	      sample_dofs_V.data(),
	      num_sample_dofs_per_procV.data(),
	      BsinvV,
	      rank,
	      nprocs,
	      numSamplesV);

  CAROM::GNAT(basisE,
	      rdime,
	      sample_dofs_E.data(),
	      num_sample_dofs_per_procE.data(),
	      BsinvE,
	      rank,
	      nprocs,
	      numSamplesE);

  // We assume that the same H1 fespace is used for X and V, and a different L2 fespace is used for E.
  // We merge all sample DOF's for X, V, and E into one set for each process.
  // The pair of spaces (H1, L2) is used here.

  vector<int> sample_dofs_merged;
  vector<int> num_sample_dofs_per_proc_merged(nprocs);
  int os_merged = 0;
  for (int p=0; p<nprocs; ++p)
    {
      std::set<int> sample_dofs_H1, sample_dofs_L2;
      {
	int os = 0;
	for (int q=0; q<p; ++q)
	  {
	    os += num_sample_dofs_per_procX[q];
	  }
	    
	for (int j=0; j<num_sample_dofs_per_procX[p]; ++j)
	  {
	    sample_dofs_H1.insert(sample_dofs_X[os + j]);
	  }

	os = 0;
	for (int q=0; q<p; ++q)
	  {
	    os += num_sample_dofs_per_procV[q];
	  }
	    
	for (int j=0; j<num_sample_dofs_per_procV[p]; ++j)
	  {
	    sample_dofs_H1.insert(sample_dofs_V[os + j]);
	  }
	    
	os = 0;
	for (int q=0; q<p; ++q)
	  {
	    os += num_sample_dofs_per_procE[q];
	  }
	    
	for (int j=0; j<num_sample_dofs_per_procE[p]; ++j)
	  {
	    sample_dofs_L2.insert(sample_dofs_E[os + j]);
	  }
      }

      num_sample_dofs_per_proc_merged[p] = sample_dofs_H1.size() + sample_dofs_L2.size();

      for (std::set<int>::const_iterator it = sample_dofs_H1.begin(); it != sample_dofs_H1.end(); ++it)
	{
	  sample_dofs_merged.push_back((*it));
	}

      for (std::set<int>::const_iterator it = sample_dofs_L2.begin(); it != sample_dofs_L2.end(); ++it)
	{
	  sample_dofs_merged.push_back(nH1[p] + (*it));  // offset by nH1[p] for the mixed spaces (H1, L2)
	}

      // For each of the num_sample_dofs_per_procX[p] samples, set s2sp_X[] to be its index in sample_dofs_merged.
      {
	int os = 0;
	for (int q=0; q<p; ++q)
	  os += num_sample_dofs_per_procX[q];

	s2sp_X.resize(numSamplesX);
	    
	for (int j=0; j<num_sample_dofs_per_procX[p]; ++j)
	  {
	    const int sample = sample_dofs_X[os + j];
		      
	    // Note: this has quadratic complexity and could be improved with a std::map<int, int>, but it should not be a bottleneck.
	    int k = -1;
	    int cnt = 0;
	    for (std::set<int>::const_iterator it = sample_dofs_H1.begin(); it != sample_dofs_H1.end(); ++it, ++cnt)
	      {
		if (*it == sample)
		  {
		    MFEM_VERIFY(k == -1, "");
		    k = cnt;
		  }
	      }

	    MFEM_VERIFY(k >= 0, "");
	    s2sp_X[os + j] = os_merged + k;
	  }
      }

      os_merged += num_sample_dofs_per_proc_merged[p];
    }
      
  // Define a superfluous finite element space, merely to get global vertex indices for the sample mesh construction.
  const int dim = pmesh->Dimension();
  H1_FECollection h1_coll(1, dim);  // Must be first order, to get a bijection between vertices and DOF's.
  ParFiniteElementSpace H1_space(pmesh, &h1_coll);  // This constructor effectively sets vertex (DOF) global indices.

  ParFiniteElementSpace *sp_H1_space, *sp_L2_space;

  MPI_Comm rom_com;
  int color = (rank != 0);
  const int status = MPI_Comm_split(MPI_COMM_WORLD, color, rank, &rom_com);
  MFEM_VERIFY(status == MPI_SUCCESS,
	      "Construction of hyperreduction comm failed");

  vector<int> sprows;
  vector<int> all_sprows;

  vector<int> s2sp;   // mapping from sample dofs in original mesh (s) to stencil dofs in sample mesh (s+), for both F and E

  // Construct sample mesh

  // This creates rom_sample_pmesh, sp_F_space, and sp_E_space only on rank 0.
  CreateSampleMesh(*pmesh, H1_space, *H1FESpace, *L2FESpace, *(H1FESpace->FEColl()),
		   *(L2FESpace->FEColl()), rom_com, sample_dofs_merged,
		   num_sample_dofs_per_proc_merged, sample_pmesh, sprows, all_sprows, s2sp, st2sp, sp_H1_space, sp_L2_space);
}

void ROM_Basis::ReadSolutionBases()
{
  /*
  basisX = ReadBasisROM(rank, ROMBasisName::X, H1size, (staticSVD ? rowOffsetH1 : 0), rdimx);
  basisV = ReadBasisROM(rank, ROMBasisName::V, H1size, (staticSVD ? rowOffsetH1 : 0), rdimv);
  basisE = ReadBasisROM(rank, ROMBasisName::E, L2size, (staticSVD ? rowOffsetL2 : 0), rdime);
  */
  
  basisX = ReadBasisROM(rank, ROMBasisName::X, tH1size, 0, rdimx);
  basisV = ReadBasisROM(rank, ROMBasisName::V, tH1size, 0, rdimv);
  basisE = ReadBasisROM(rank, ROMBasisName::E, tL2size, 0, rdime);
}

// f is a full vector, not a true vector
void ROM_Basis::ProjectFOMtoROM(Vector const& f, Vector & r)
{
  MFEM_VERIFY(r.Size() == rdimx + rdimv + rdime, "");
  MFEM_VERIFY(f.Size() == (2*H1size) + L2size, "");

  for (int i=0; i<H1size; ++i)
    gfH1[i] = f[i];

  gfH1.GetTrueDofs(mfH1);
    
  for (int i=0; i<tH1size; ++i)
    (*fH1)(i) = mfH1[i];
  
  basisX->transposeMult(*fH1, *rX);

  for (int i=0; i<H1size; ++i)
    gfH1[i] = f[H1size + i];

  gfH1.GetTrueDofs(mfH1);
    
  for (int i=0; i<tH1size; ++i)
    (*fH1)(i) = mfH1[i];
  
  basisV->transposeMult(*fH1, *rV);
  
  for (int i=0; i<L2size; ++i)
    gfL2[i] = f[(2*H1size) + i];

  gfL2.GetTrueDofs(mfL2);
    
  for (int i=0; i<tL2size; ++i)
    (*fL2)(i) = mfL2[i];
  
  basisE->transposeMult(*fL2, *rE);
  
  for (int i=0; i<rdimx; ++i)
    r[i] = (*rX)(i);

  for (int i=0; i<rdimv; ++i)
    r[rdimx + i] = (*rV)(i);

  for (int i=0; i<rdime; ++i)
    r[rdimx + rdimv + i] = (*rE)(i);
}

// f is a full vector, not a true vector
void ROM_Basis::LiftROMtoFOM(Vector const& r, Vector & f)
{
  MFEM_VERIFY(r.Size() == rdimx + rdimv + rdime, "");
  MFEM_VERIFY(f.Size() == (2*H1size) + L2size, "");

  for (int i=0; i<rdimx; ++i)
    (*rX)(i) = r[i];

  for (int i=0; i<rdimv; ++i)
    (*rV)(i) = r[rdimx + i];

  for (int i=0; i<rdime; ++i)
    (*rE)(i) = r[rdimx + rdimv + i];

  basisX->mult(*rX, *fH1);

  for (int i=0; i<tH1size; ++i)
    mfH1[i] = (*fH1)(i);

  gfH1.SetFromTrueDofs(mfH1);
  
  for (int i=0; i<H1size; ++i)
    f[i] = gfH1[i];
  
  basisV->mult(*rV, *fH1);

  for (int i=0; i<tH1size; ++i)
    mfH1[i] = (*fH1)(i);

  gfH1.SetFromTrueDofs(mfH1);
  
  for (int i=0; i<H1size; ++i)
    f[H1size + i] = gfH1[i];

  basisE->mult(*rE, *fL2);

  for (int i=0; i<tL2size; ++i)
    mfL2[i] = (*fL2)(i);

  gfL2.SetFromTrueDofs(mfL2);
  
  for (int i=0; i<L2size; ++i)
    f[(2*H1size) + i] = gfL2[i];
}

ROM_Operator::ROM_Operator(hydrodynamics::LagrangianHydroOperator *lhoper, ROM_Basis *b,
			   FunctionCoefficient& rho_coeff, FunctionCoefficient& mat_coeff,
			   const int order_e, const int source, const bool visc, const double cfl,
			   const double cg_tol, const double ftz_tol, const bool hyperreduce_,
			   H1_FECollection *H1fec, FiniteElementCollection *L2fec)
  : TimeDependentOperator(b->TotalSize()), operFOM(lhoper), basis(b),
    fx(lhoper->Height()), fy(lhoper->Height()), hyperreduce(hyperreduce_)
{
  MFEM_VERIFY(lhoper->Height() == lhoper->Width(), "");

  if (hyperreduce)
    {
      spmesh = b->GetSampleMesh();

      // The following code is copied from laghos.cpp to define a LagrangianHydroOperator on spmesh.
      
      L2FESpaceSP = new ParFiniteElementSpace(spmesh, L2fec);
      H1FESpaceSP = new ParFiniteElementSpace(spmesh, H1fec, spmesh->Dimension());
      
      Vsize_l2sp = L2FESpaceSP->GetVSize();
      Vsize_h1sp = H1FESpaceSP->GetVSize();

      Array<int> ossp(4);
      ossp[0] = 0;
      ossp[1] = ossp[0] + Vsize_h1sp;
      ossp[2] = ossp[1] + Vsize_h1sp;
      ossp[3] = ossp[2] + Vsize_l2sp;
      BlockVector S(ossp);

      // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
      // that the boundaries are straight.
      Array<int> ess_tdofs;
      {
	Array<int> ess_bdr(spmesh->bdr_attributes.Max()), tdofs1d;
	for (int d = 0; d < spmesh->Dimension(); d++)
	  {
	    // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
	    // enforce v_x/y/z = 0 for the velocity components.
	    ess_bdr = 0; ess_bdr[d] = 1;
	    H1FESpaceSP->GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
	    ess_tdofs.Append(tdofs1d);
	  }
      }

      ParGridFunction rho(L2FESpaceSP);
      L2_FECollection l2_fec(order_e, spmesh->Dimension());
      ParFiniteElementSpace l2_fes(spmesh, &l2_fec);
      ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
      l2_rho.ProjectCoefficient(rho_coeff);
      rho.ProjectGridFunction(l2_rho);

      // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
      // gamma values are projected on a function that stays constant on the moving
      // mesh.
      L2_FECollection mat_fec(0, spmesh->Dimension());
      ParFiniteElementSpace mat_fes(spmesh, &mat_fec);
      ParGridFunction mat_gf(&mat_fes);
      mat_gf.ProjectCoefficient(mat_coeff);
      GridFunctionCoefficient *mat_gf_coeff = new GridFunctionCoefficient(&mat_gf);

      const bool p_assembly = false;
      const int cg_max_iter = 300;

      operSP = new hydrodynamics::LagrangianHydroOperator(S.Size(), *H1FESpaceSP, *L2FESpaceSP,
							  ess_tdofs, rho, source, cfl, mat_gf_coeff,
							  visc, p_assembly, cg_tol, cg_max_iter, ftz_tol,
							  H1fec->GetBasisType());
    }
}

void ROM_Operator::Mult(const Vector &x, Vector &y) const
{
  if (hyperreduce)
    {
      MFEM_VERIFY(false, "Hyperreduction not implemented yet in ROM_Operator::Mult");
    }
  else
    {
      basis->LiftROMtoFOM(x, fx);
      operFOM->Mult(fx, fy);
      basis->ProjectFOMtoROM(fy, y);
    }
}

void PrintL2NormsOfParGridFunctions(const int rank, const std::string& name, ParGridFunction *f1, ParGridFunction *f2,
				    const bool scalar)
{
  ConstantCoefficient zero(0.0);
  Vector zerov(3);
  zerov = 0.0;
  VectorConstantCoefficient vzero(zerov);

  double fomloc, romloc, diffloc;

  // TODO: why does ComputeL2Error call the local GridFunction version rather than the global ParGridFunction version?
  // Only f2->ComputeL2Error calls the ParGridFunction version.
  if (scalar)
    {
      fomloc = f1->ComputeL2Error(zero);
      romloc = f2->ComputeL2Error(zero);
    }
  else
    {
      fomloc = f1->ComputeL2Error(vzero);
      romloc = f2->ComputeL2Error(vzero);
    }
    
  *f1 -= *f2;  // works because GridFunction is derived from Vector

  if (scalar)
    {
      diffloc = f1->ComputeL2Error(zero);
    }
  else
    {
      diffloc = f1->ComputeL2Error(vzero);
    }
  
  double fomloc2 = fomloc*fomloc;
  double romloc2 = romloc*romloc;
  double diffloc2 = diffloc*diffloc;
	  
  double fomglob2, romglob2, diffglob2;

  // TODO: is this right? The "loc" norms should be global, but they are not.
  MPI_Allreduce(&fomloc2, &fomglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&romloc2, &romglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&diffloc2, &diffglob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /*
  fomglob2 = fomloc2;
  romglob2 = romloc2;
  diffglob2 = diffloc2;
  */
  
  cout << rank << ": " << name << " FOM norm " << sqrt(fomglob2) << endl;
  cout << rank << ": " << name << " ROM norm " << sqrt(romglob2) << endl;
  cout << rank << ": " << name << " DIFF norm " << sqrt(diffglob2) << endl;
}