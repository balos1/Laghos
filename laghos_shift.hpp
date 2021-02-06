// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_LAGHOS_SHIFT
#define MFEM_LAGHOS_SHIFT

#include "mfem.hpp"

namespace mfem
{

namespace hydrodynamics
{

int material_id(int el_id, const ParGridFunction &g);

double interfaceLS(const Vector &x);

void MarkFaceAttributes(ParMesh &pmesh);

// Performs full assemble for the force face terms:
// F_face_ij = - int_face [ [grad_p * dist] * h1_shape_j l2_shape_i].
class FaceForceIntegrator : public BilinearFormIntegrator
{
private:
   Vector h1_shape_face, l2_shape;
   const ParGridFunction &p;
   VectorCoefficient &dist;

  public:
   FaceForceIntegrator(const ParGridFunction &p_gf,
                       VectorCoefficient &d) : p(p_gf), dist(d)  { }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   void AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                           const FiniteElement &test_fe1,
                           const FiniteElement &test_fe2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat);
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_LAGHOS_TMOP