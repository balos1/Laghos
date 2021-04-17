// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project (17-SC-20-SC)
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "laghos_shift.hpp"
#include "laghos_solver.hpp"

using namespace std;
namespace mfem
{

namespace hydrodynamics
{

int material_id(int el_id, const ParGridFunction &g)
{
   const ParFiniteElementSpace &pfes =  *g.ParFESpace();
   const FiniteElement *fe = pfes.GetFE(el_id);
   Vector g_vals;
   const IntegrationRule &ir =
      IntRules.Get(fe->GetGeomType(), pfes.GetOrder(el_id) + 7);

   double integral = 0.0;
   bool is_positive = true;
   g.GetValues(el_id, ir, g_vals);
   ElementTransformation *Tr = pfes.GetMesh()->GetElementTransformation(el_id);
   for (int q = 0; q < ir.GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir.IntPoint(q);
      Tr->SetIntPoint(&ip);
      integral += ip.weight * g_vals(q) * Tr->Weight();
      if (g_vals(q) + 1e-12 < 0.0) { is_positive = false; }
   }
   return (is_positive) ? 1 : 0;
   //return (integral > 0.0) ? 1 : 0;
}

double interfaceLS(const Vector &x)
{
   // 0 - vertical - //sod
   // 1 - diagonal
   // 2 - circle
   // 3 - vertical at 0.7 (water-air test).
   const int mode = 3;

   const int dim = x.Size();
   switch (mode)
   {
      case 0: return tanh(x(0) - 0.5);
      case 1: return tanh(x(0) - x(1));
      case 2:
      {
      double center[3] = {0.5, 0.5, 0.5};
         double rad = 0.0;
         for (int d = 0; d < dim; d++)
         {
            rad += (x(d) - center[d]) * (x(d) - center[d]);
         }
         rad = sqrt(rad + 1e-16);
         return tanh(rad - 0.3);
      }
      case 3: return tanh(x(0) - 0.7);
      default: MFEM_ABORT("error"); return 0.0;
   }
}

void MarkFaceAttributes(ParMesh &pmesh)
{
   // Set face_attribute = 77 to faces that are on the material interface.
   for (int f = 0; f < pmesh.GetNumFaces(); f++)
   {
      auto *ftr = pmesh.GetFaceElementTransformations(f, 3);
      if (ftr->Elem2No > 0 &&
          pmesh.GetAttribute(ftr->Elem1No) != pmesh.GetAttribute(ftr->Elem2No))
      {
         pmesh.SetFaceAttribute(f, 77);
      }
   }
}


void GradAtLocalDofs(ElementTransformation &T,
                     const ParGridFunction &g,
                     DenseMatrix &grad_g){
   ParFiniteElementSpace &pfes = *g.ParFESpace();
   const FiniteElement &el = *pfes.GetFE(T.ElementNo);
   const int dim = el.GetDim(), dof = el.GetDof();
   grad_g.SetSize(dof, dim);
   Array<int> dofs;
   Vector g_e;
   DenseMatrix grad_phys; // This will be (dof_p x dim, dof_p).
   {
      pfes.GetElementDofs(T.ElementNo, dofs);
      g.GetSubVector(dofs, g_e);
      el.ProjectGrad(el, T, grad_phys);
      Vector grad_ptr(grad_g.GetData(), dof*dim);
      grad_phys.Mult(g_e, grad_ptr);
   }
}

void StrainTensorAtLocalDofs(ElementTransformation &T, const ParGridFunction &g,
                             DenseTensor &grad_g)
{
   ParFiniteElementSpace &pfes = *g.ParFESpace();
   const int zone_id = T.ElementNo;
   const FiniteElement &el = *pfes.GetFE(zone_id);
   const int dim = el.GetDim(), dof = el.GetDof();
   MFEM_VERIFY(dim == pfes.GetVDim(), " Strain Tensor can only be obtained for"
                                      " vector GridFunctions.");
   grad_g.SetSize(dof, dim, dim);
   Array<int> dofs;
   Vector g_e;
   DenseMatrix grad_phys; // This will be (dof_p x dim, dof_p).
   {
      pfes.GetElementVDofs(zone_id, dofs);
      g.GetSubVector(dofs, g_e);
      el.ProjectGrad(el, T, grad_phys);
      for (int d = 0; d < dim; d++) {
          Vector g_e_d(g_e.GetData()+d*dof, dof);
          Vector grad_ptr(grad_g.GetData(0)+d*dof*dim, dof*dim);
          grad_phys.Mult(g_e_d, grad_ptr);
//          DenseMatrix grad_g_slice(grad_g.GetData(d), dof, dim);
//          std::cout << " print slice\n";
//          grad_g_slice.Print();
      }
   }
}

void FaceForceIntegrator::AssembleFaceMatrix(const FiniteElement &trial_face_fe,
                                             const FiniteElement &test_fe1,
                                             const FiniteElement &test_fe2,
                                             FaceElementTransformations &Trans,
                                             DenseMatrix &elmat)
{
   const int h1dofs_cnt_face = trial_face_fe.GetDof();
   const int l2dofs_cnt = test_fe1.GetDof();
   const int dim = test_fe1.GetDim();

   if (Trans.Elem2No < 0)
   {
      // This case should take care of shared (MPI) faces. They will get
      // processed by both MPI tasks.
      elmat.SetSize(l2dofs_cnt, h1dofs_cnt_face * dim);
   }
   elmat.SetSize(l2dofs_cnt * 2, h1dofs_cnt_face * dim);
   elmat = 0.0;

   // Must be done after elmat.SetSize().
   if (Trans.Attribute != 77) { return; }

   h1_shape_face.SetSize(h1dofs_cnt_face);
   l2_shape.SetSize(l2dofs_cnt);

   const int ir_order =
      test_fe1.GetOrder() + trial_face_fe.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all quad points, on both sides.
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof();
   DenseMatrix p_grad_e_1(dof_p, dim), p_grad_e_2(dof_p, dim);
   GradAtLocalDofs(Trans.GetElement1Transformation(), p, p_grad_e_1);
   if (Trans.Elem2No > 0)
   {
      GradAtLocalDofs(Trans.GetElement2Transformation(), p, p_grad_e_2);
   }

   Vector nor(dim);

   Vector p_grad_q(dim), d_q(dim), shape_p(dof_p);
   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }
      nor *= ip_f.weight;

      // Shape functions on the face (H1); same for both elements.
      trial_face_fe.CalcShape(ip_f, h1_shape_face);

      const double p1 = p.GetValue(Trans.GetElement1Transformation(), ip_e1);
      const double p2 = p.GetValue(Trans.GetElement2Transformation(), ip_e2);

      double grad_p_d_jump = 0.;
      Vector grad_p_d_nor(dim);
      Array<int> dofs;

      // 1st element.
      {
         // Compute dist * grad_p in the first element.
         el_p.CalcShape(ip_e1, shape_p);
         p_grad_e_1.MultTranspose(shape_p, p_grad_q);
         dist.Eval(d_q, Trans.GetElement1Transformation(), ip_e1);
         const double grad_p_d = d_q * p_grad_q;

         // L2 shape functions in the 1st element.
         test_fe1.CalcShape(ip_e1, l2_shape);

         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt_face; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  elmat(i, d*h1dofs_cnt_face + j)
                          += grad_p_d * l2_shape(i) * h1_shape_face(j) * nor(d);
               }
            }
         }
      }
      // 2nd element if there is such (subtracting from the 1st).
      if (Trans.Elem2No >= 0)
      {
         // Compute dist * grad_p in the second element.
         el_p.CalcShape(ip_e2, shape_p);
         p_grad_e_2.MultTranspose(shape_p, p_grad_q);
         dist.Eval(d_q, Trans.GetElement2Transformation(), ip_e2);
         const double grad_p_d = d_q * p_grad_q;

         // L2 shape functions on the 2nd element.
         test_fe2.CalcShape(ip_e2, l2_shape);

         for (int i = 0; i < l2dofs_cnt; i++)
         {
            for (int j = 0; j < h1dofs_cnt_face; j++)
            {
               for (int d = 0; d < dim; d++)
               {
                  elmat(l2dofs_cnt + i, d*h1dofs_cnt_face + j)
                          -= grad_p_d * l2_shape(i) * h1_shape_face(j) * nor(d);
               }
            }
         }
      }
   }
}

void EnergyInterfaceIntegrator::AssembleRHSFaceVect(const FiniteElement &el_1,
                                                    const FiniteElement &el_2,
                                                    FaceElementTransformations &Trans,
                                                    Vector &elvect)
{
   const int l2dofs_cnt = el_1.GetDof();
   const int dim = el_1.GetDim();

   if (Trans.Elem2No < 0)
   {
      // This case should take care of shared (MPI) faces. They will get
      // processed by both MPI tasks.
      elvect.SetSize(l2dofs_cnt);
   }
   elvect.SetSize(l2dofs_cnt * 2);
   elvect = 0.0;

   // Must be done after elvect.SetSize().
   if (Trans.Attribute != 77) { return; }

   Vector l2_shape(l2dofs_cnt);

   const int ir_order =
      el_1.GetOrder() + Trans.OrderW();
   const IntegrationRule *ir = &IntRules.Get(Trans.GetGeometryType(), ir_order);
   const int nqp_face = ir->GetNPoints();

   // grad_p at all quad points, on both sides.
   Vector p_e;
   Array<int> dofs_p;
   const FiniteElement &el_p = *p.ParFESpace()->GetFE(0);
   const FiniteElement &el_v = *v.ParFESpace()->GetFE(0);
   const int dof_p = el_p.GetDof(), dof_v = el_v.GetDof();
   DenseMatrix p_grad_e_1(dof_p, dim), p_grad_e_2(dof_p, dim);
   if (Trans.Elem2No > 0)
   {
      GradAtLocalDofs(Trans.GetElement2Transformation(), p, p_grad_e_2);
   }
   GradAtLocalDofs(Trans.GetElement1Transformation(), p, p_grad_e_1);

   Vector nor(dim);

   Vector p_grad_q1(dim), p_grad_q2(dim), d_q1(dim), d_q2(dim),
         shape_p(dof_p), v_vals(dim), shape_v(dof_v);

   DenseTensor v_strain_e_1(dof_v, dim, dim), v_strain_e_2(dof_v, dim, dim);
   if (Trans.Elem2No > 0)
   {
       StrainTensorAtLocalDofs(Trans.GetElement2Transformation(), v, v_strain_e_2);
   }
   StrainTensorAtLocalDofs(Trans.GetElement1Transformation(), v, v_strain_e_1);

   DenseMatrix v_strain_q1(dim), v_strain_q2(dim);

   for (int q = 0; q  < nqp_face; q++)
   {
      const IntegrationPoint &ip_f = ir->IntPoint(q);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip_f);

      // Access the neighboring elements' integration points
      // Note: eip2 will only contain valid data if Elem2 exists
      const IntegrationPoint &ip_e1 = Trans.GetElement1IntPoint();
      const IntegrationPoint &ip_e2 = Trans.GetElement2IntPoint();

      // The normal includes the Jac scaling.
      // The orientation is taken into account in the processing of element 1.
      if (dim == 1)
      {
         nor(0) = (2*ip_e1.x - 1.0 ) * Trans.Weight();
      }
      else { CalcOrtho(Trans.Jacobian(), nor); }
      nor *= ip_f.weight;

      // 1st element stuff.
      el_p.CalcShape(ip_e1, shape_p);
      const double p1 = p.GetValue(Trans.GetElement1Transformation(), ip_e1);
      dist.Eval(d_q1, Trans.GetElement1Transformation(), ip_e1);
      p_grad_e_1.MultTranspose(shape_p, p_grad_q1);

      el_v.CalcShape(ip_e1, shape_v);
      for (int d = 0; d < dim; d++) {
          DenseMatrix v_grad_e_1(v_strain_e_1.GetData(d), dof_v, dim);
          Vector v_grad_q1(v_strain_q1.GetData()+d*dim, dim);
          v_grad_e_1.MultTranspose(shape_v, v_grad_q1);
      }
      v_strain_q1.Transpose(); //du_i/dx_j

      // 2nd element stuff.
      el_p.CalcShape(ip_e2, shape_p);
      const double p2 = p.GetValue(Trans.GetElement2Transformation(), ip_e2);
      dist.Eval(d_q2, Trans.GetElement2Transformation(), ip_e2);
      p_grad_e_2.MultTranspose(shape_p, p_grad_q2);

      el_v.CalcShape(ip_e2, shape_v);
      for (int d = 0; d < dim; d++) {
          DenseMatrix v_grad_e_2(v_strain_e_2.GetData(d), dof_v, dim);
          Vector v_grad_q2(v_strain_q2.GetData()+d*dim, dim);
          v_grad_e_2.MultTranspose(shape_v, v_grad_q2);
      }
      v_strain_q2.Transpose();

      v.GetVectorValue(Trans, ip_f, v_vals);
      double p_jump_term = 0.0;
      p_jump_term = -0.5*(p1 + d_q1 * p_grad_q1 - p2 - d_q2 * p_grad_q2) *
                     (nor * v_vals); //0.5 for {{phi}}

      // 1st element.
      {
         // L2 shape functions in the 1st element.
         el_1.CalcShape(ip_e1, l2_shape);
         for (int i = 0; i < l2dofs_cnt; i++)
         {
             elvect(i) += l2_shape(i) * p_jump_term;
         }
      }
      // 2nd element.
      {
         // L2 shape functions in the 2nd element.
         el_2.CalcShape(ip_e2, l2_shape);
         for (int i = 0; i < l2dofs_cnt; i++)
         {
             elvect(i + l2dofs_cnt) += l2_shape(i) * p_jump_term;
         }
      }

      double gradv_d_n_n_jump = 0.;
      double gradp_d_jump = 0.0;
      double p_avg = 0.0;
      {
          Vector gradv_d(dim);
          v_strain_q1.Mult(d_q1, gradv_d);
          Vector true_normal = d_q1;
          true_normal *= d_q1.Norml2() == 0. ? 0 : 1./d_q1.Norml2();
          double gradv_d_n = gradv_d*true_normal;
          Vector gradv_d_n_n1 = true_normal;
          gradv_d_n_n1 *= gradv_d_n;

          v_strain_q2.Mult(d_q2, gradv_d);
          true_normal = d_q2;
          true_normal *= d_q2.Norml2() == 0. ? 0 : 1./d_q2.Norml2();
          gradv_d_n = gradv_d*true_normal;
          Vector gradv_d_n_n2 = true_normal;
          gradv_d_n_n2 *= gradv_d_n;

          gradv_d_n_n_jump = gradv_d_n_n1*nor - gradv_d_n_n2*nor;
          gradp_d_jump = p_grad_q1*d_q1 - p_grad_q2*d_q2;
          p_avg = 0.5*(p1 + p2);
      }

      // 1st element.
      {
         // L2 shape functions in the 1st element.
         el_1.CalcShape(ip_e1, l2_shape);

         Vector gradp_d_jump_dot_l2_shape_jump = l2_shape;
         gradp_d_jump_dot_l2_shape_jump *= gradp_d_jump;
         gradp_d_jump_dot_l2_shape_jump *= -0.25;

         Vector l2_shape_avg_p_avg = l2_shape;
         l2_shape_avg_p_avg *= 0.5;
         l2_shape_avg_p_avg *= p_avg;
         gradp_d_jump_dot_l2_shape_jump += l2_shape_avg_p_avg;
         gradp_d_jump_dot_l2_shape_jump *= gradv_d_n_n_jump;
         Vector elvect_temp(elvect.GetData(), l2dofs_cnt);
         elvect_temp.Add(1., gradp_d_jump_dot_l2_shape_jump);
      }

      // 2nd element
      {
          el_2.CalcShape(ip_e2, l2_shape);

          Vector gradp_d_jump_dot_l2_shape_jump = l2_shape;
          l2_shape *= -1; //l2_shape_jump = phi+ - phi- = -phi-
          gradp_d_jump_dot_l2_shape_jump *= gradp_d_jump;
          gradp_d_jump_dot_l2_shape_jump *= -0.25;

          Vector l2_shape_avg_p_avg = l2_shape;
          l2_shape_avg_p_avg *= 0.5;
          l2_shape_avg_p_avg *= p_avg;
          gradp_d_jump_dot_l2_shape_jump += l2_shape_avg_p_avg;
          gradp_d_jump_dot_l2_shape_jump *= gradv_d_n_n_jump;
          Vector elvect_temp(elvect.GetData() + l2dofs_cnt, l2dofs_cnt);
          elvect_temp.Add(1., gradp_d_jump_dot_l2_shape_jump);
      }
   }
}

int FindPointDOF(const int z_id, const Vector &xyz,
                 const ParFiniteElementSpace &pfes)
{
   const IntegrationRule &ir = pfes.GetFE(z_id)->GetNodes();
   const int dofs_cnt = ir.GetNPoints(), dim = pfes.GetParMesh()->Dimension();
   ElementTransformation &tr = *pfes.GetElementTransformation(z_id);
   Vector position;
   Array<int> dofs;
   const double eps = 1e-8;
   for (int j = 0; j < dofs_cnt; j++)
   {
      pfes.GetElementDofs(z_id, dofs);
      const IntegrationPoint &ip = ir.IntPoint(j);
      tr.SetIntPoint(&ip);
      tr.Transform(ip, position);
      bool found = true;
      for (int d = 0; d < dim; d++)
      {
         //std::cout << dofs[j] << " " << position(d) << " " << xyz(d) << std::endl;
         if (fabs(position(d) - xyz(d)) > eps) { found = false; break; }
      }
      if (found)
      {
         return dofs[j];
      }
   }
   return -1;
}

void PrintCellNumbers(const Vector &xyz, const ParFiniteElementSpace &pfes)
{
   MFEM_VERIFY(pfes.GetNRanks() == 1, "PointExtractor works only in serial.");

   const int NE = pfes.GetNE();
   int dof_id;
   for (int i = 0; i < NE; i++)
   {
      dof_id = FindPointDOF(i, xyz, pfes);
      if (dof_id > 0)
      {
         std::cout << "Element " << i << "; Dof: " << dof_id << endl;
      }
   }
}

PointExtractor::PointExtractor(int z_id, Vector &xyz,
                               const ParGridFunction &gf,
                               std::string filename)
   : g(gf), dof_id(-1), fstream(filename)
{
   ParFiniteElementSpace &pfes = *gf.ParFESpace();
   MFEM_VERIFY(pfes.GetNRanks() == 1, "PointExtractor works only in serial.");

   dof_id = FindPointDOF(z_id, xyz, pfes);
   MFEM_VERIFY(dof_id > -1,
               "Wrong zone specification for extraction " << filename);

   fstream.precision(8);
}

void PointExtractor::WriteValue(double time)
{
   fstream << time << " " << GetValue() << "\n";
   fstream.flush();
}

ShiftedPointExtractor::ShiftedPointExtractor(int z_id, Vector &xyz,
                                             const ParGridFunction &gf,
                                             const ParGridFunction &d,
                                             string filename)
   : PointExtractor(z_id, xyz, gf, filename),
     dist(d), zone_id(z_id), dist_dof_id(-1)
{
   ParFiniteElementSpace &pfes = *dist.ParFESpace();
   MFEM_VERIFY(pfes.GetNRanks() == 1,
               "ShiftedPointExtractor works only in serial.");

   dist_dof_id = FindPointDOF(z_id, xyz, pfes);
   MFEM_VERIFY(dist_dof_id > -1,
               "Wrong zone specification for extraction (distance field).");
}

double ShiftedPointExtractor::GetValue() const
{
   ParFiniteElementSpace &pfes = *g.ParFESpace();
   const FiniteElement &el = *pfes.GetFE(zone_id);
   const int dim = el.GetDim(), dof = el.GetDof();

   // Gradient of the field at the point.
   DenseMatrix grad_e;
   GradAtLocalDofs(*pfes.GetElementTransformation(zone_id), g, grad_e);

   Array<int> dofs;
   pfes.GetElementDofs(zone_id, dofs);
   int loc_dof_id = -1;
   for (int i = 0; i < dof; i++)
   {
      if (dofs[i] == dof_id) { loc_dof_id = i; break; }
   }
   MFEM_VERIFY(loc_dof_id >= 0, "Can't find the dof in the zone!");

   double res = g(dof_id);
   const int dsize = dist.Size();
   for (int d = 0; d < dim; d++)
   {
      res += dist(dsize*d + dist_dof_id) * grad_e(loc_dof_id, d);
   }

   return res;

}

void InitSod2Mat(ParGridFunction &rho, ParGridFunction &v,
                 ParGridFunction &e, ParGridFunction &gamma)
{
   v = 0.0;

   ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = rho.Size() / NE;
   double r, g, p;
   for (int i = 0; i < NE; i++)
   {
      if (pfes.GetParMesh()->GetAttribute(i) == 1)
      {
         r = 1.000; g = 2.0; p = 2.0;
      }
      else
      {
         r = 0.125; g = 1.4; p = 0.1;
      }

      gamma(i) = g;
      for (int j = 0; j < ndofs; j++)
      {
         rho(i*ndofs + j) = r;
         e(i*ndofs + j)   = p / r / (g - 1.0);
      }
   }
}

void InitWaterAir(ParGridFunction &rho, ParGridFunction &v,
                  ParGridFunction &e, ParGridFunction &gamma)
{
   v = 0.0;

   ParFiniteElementSpace &pfes = *rho.ParFESpace();
   const int NE    = pfes.GetNE();
   const int ndofs = rho.Size() / NE;
   double r, g, p;
   for (int i = 0; i < NE; i++)
   {
      if (pfes.GetParMesh()->GetAttribute(i) == 1)
      {
         r = 1000; g = 4.4; p = 1.e9;
         double A = 6.0e8;
         gamma(i) = g;
         for (int j = 0; j < ndofs; j++)
         {
            rho(i*ndofs + j) = r;
            e(i*ndofs + j)   = (p + g*A) / r / (g - 1.0);
         }
      }
      else
      {
         r = 50; g = 1.4; p = 1.e5;
         gamma(i) = g;
         for (int j = 0; j < ndofs; j++)
         {
            rho(i*ndofs + j) = r;
            e(i*ndofs + j)   = p / r / (g - 1.0);
         }
      }
   }
}

} // namespace hydrodynamics

} // namespace mfem
