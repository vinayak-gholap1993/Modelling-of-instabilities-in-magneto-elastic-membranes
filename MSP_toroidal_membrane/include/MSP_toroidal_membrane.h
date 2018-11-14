#ifndef MSP_TOROIDAL_MEMBRANE_H
#define MSP_TOROIDAL_MEMBRANE_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/path_search.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/bounding_box.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_postprocessor.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
// For block system:
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <mpi.h>

using namespace dealii;

// @sect3{Run-time parameters}
//
// There are several parameters that can be set in the code so we set up a
// ParameterHandler object to read in the choices at run-time.
namespace Parameters
{
// User input parameters for the load stepping
struct LoadStep
{
  double total_load;
  double delta_load;

  static void
  declare_parameters(ParameterHandler &prm);

  void
  parse_parameters(ParameterHandler &prm);
};

void LoadStep::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Load step");
    {
        prm.declare_entry("Total load", "-0.1",
                          Patterns::Double(),
                          "Final load value at the end of all load steps");

        prm.declare_entry("Load step size", "-0.05",
                          Patterns::Double(),
                          "Load step size for each load increment cylce");
    }
    prm.leave_subsection();
}

void LoadStep::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Load step");
    {
        total_load = prm.get_double("Total load");
        delta_load = prm.get_double("Load step size");
    }
    prm.leave_subsection();
}

// User input for the problem type to solve for
// Can be purely magnetic problem for magnetic scalar potential
// Can be purely elastic problem for vector valued displacement in finite strain elasticity
// Can be a coupled problem involving magnetic field and vector valued displacement to be solved for
// as a mixed problem
  struct ProblemType
  {
      std::string problem_type;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
  };

  void ProblemType::declare_parameters(ParameterHandler &prm)
  {
      prm.enter_subsection("Problem type");
      {
          prm.declare_entry("Problem type", "Purely magnetic",
                            Patterns::Selection("Purely magnetic | Purely elastic | Coupled magnetoelastic"),
                            "The problem type to solve");
      }
      prm.leave_subsection();
  }

  void ProblemType::parse_parameters(ParameterHandler &prm)
  {
      prm.enter_subsection("Problem type");
      {
          problem_type = prm.get("Problem type");
      }
      prm.leave_subsection();
  }

// @sect4{Boundary conditions}

  struct BoundaryConditions
  {
    double potential_difference_per_unit_length;
    std::string mechanical_boundary_condition_type; // for deformation of the body by mechanical load
    double prescribed_traction_load; // for Traction B.C.

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };


  void BoundaryConditions::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Boundary conditions");
    {
      prm.declare_entry("Potential difference per unit length", "1000",
                        Patterns::Double(1e-9),
                        "Potential difference along the faux permanent magnet");

      prm.declare_entry("Mechanical boundary condition type", "Inhomogeneous Dirichlet",
                        Patterns::Selection("Inhomogeneous Dirichlet | Traction"),
                        "Type of boundary condition to deform body by mechanical load");

      prm.declare_entry("Prescribed traction load", "1.0e-3",
                        Patterns::Double(),
                        "Traction force per unit area");

    }
    prm.leave_subsection();
  }

  void BoundaryConditions::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Boundary conditions");
    {
      potential_difference_per_unit_length = prm.get_double("Potential difference per unit length");
      mechanical_boundary_condition_type = prm.get("Mechanical boundary condition type");
      prescribed_traction_load = prm.get_double("Prescribed traction load");
    }
    prm.leave_subsection();
  }

// @sect4{Finite Element system}

// As mentioned in the introduction, a different order interpolation should be
// used for the displacement $\mathbf{u}$ than for the pressure
// $\widetilde{p}$ and the dilatation $\widetilde{J}$.  Choosing
// $\widetilde{p}$ and $\widetilde{J}$ as discontinuous (constant) functions
// at the element level leads to the mean-dilatation method. The discontinuous
// approximation allows $\widetilde{p}$ and $\widetilde{J}$ to be condensed
// out and a classical displacement based method is recovered.  Here we
// specify the polynomial order used to approximate the solution.  The
// quadrature order should be adjusted accordingly, but this is done at a
// later stage.
  struct FESystem
  {
    unsigned int poly_degree_min;
    unsigned int poly_degree_max;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };


  void FESystem::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Finite element system");
    {
      prm.declare_entry("Minimum polynomial degree", "1",
                        Patterns::Integer(1),
                        "Displacement system polynomial order");

      prm.declare_entry("Maximum polynomial degree", "2",
                        Patterns::Integer(1),
                        "Displacement system polynomial order");
    }
    prm.leave_subsection();
  }

  void FESystem::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Finite element system");
    {
      poly_degree_min = prm.get_integer("Minimum polynomial degree");
      poly_degree_max = prm.get_integer("Maximum polynomial degree");
    }
    prm.leave_subsection();
  }

// @sect4{Geometry}

// Make adjustments to the problem geometry and the applied load.  Since the
// problem modelled here is quite specific, the load scale can be altered to
// specific values to compare with the results given in the literature.
  struct Geometry
  {
    std::string mesh_file;
    double      torus_major_radius;
    double      torus_minor_radius_inner;
    double      torus_minor_radius_outer;
    double      grid_scale;
    double      bounding_box_r;
    double      bounding_box_z;
    std::string geometry_shape;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void Geometry::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      prm.declare_entry("Mesh file", "../mesh/new_2d_toroidal_membrane.inp",
                        Patterns::Anything(),
                        "Mesh file for the toroidal geometry");

      prm.declare_entry("Torus major radius", "0.5",
                        Patterns::Double(0.0),
                        "Major radius of the torus");

      prm.declare_entry("Torus minor radius (inner)", "0.195",
                        Patterns::Double(0.0),
                        "Minor inner radius of the torus");

      prm.declare_entry("Torus minor radius (outer)", "0.2",
                        Patterns::Double(0.0),
                        "Minor outer radius of the torus");

      prm.declare_entry("Grid scale", "1.0",
                        Patterns::Double(0.0),
                        "Global grid scaling factor");      

      prm.declare_entry("Magnet bounding box radial length", "0.035",
                        Patterns::Double(0.0),
                        "Permanent magnet region radial (x) length");

      prm.declare_entry("Magnet bounding box axial length", "0.10",
                        Patterns::Double(0.0),
                        "Permanent magnet region axial (z) length");

      prm.declare_entry("Geometry shape for the problem", "Toroidal_tube",
                        Patterns::Selection("Toroidal_tube | Beam | Patch test"),
                        "Geometry selection for problem");
    }
    prm.leave_subsection();
  }

  void Geometry::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      mesh_file = prm.get("Mesh file");
      torus_major_radius = prm.get_double("Torus major radius");
      torus_minor_radius_inner = prm.get_double("Torus minor radius (inner)");
      torus_minor_radius_outer = prm.get_double("Torus minor radius (outer)");
      grid_scale = prm.get_double("Grid scale");
      bounding_box_r = prm.get_double("Magnet bounding box radial length");
      bounding_box_z = prm.get_double("Magnet bounding box axial length");
      geometry_shape = prm.get("Geometry shape for the problem");
    }
    prm.leave_subsection();
  }

// @sect4{Refinement}

  struct Refinement
  {
    std::string refinement_strategy;
    unsigned int n_global_refinements;
    unsigned int n_cycles_max;
    unsigned int n_levels_max;
    double frac_refine;
    double frac_coarsen;
    double force_manifold_refinement;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };


  void Refinement::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Refinement");
    {
      prm.declare_entry("Refinement strategy", "h-AMR",
                        Patterns::Selection("h-GMR|p-GMR|h-AMR|p-AMR"), // hp-AMR
                        "Strategy used to perform hp refinement");

      prm.declare_entry("Initial global refinements", "1",
                        Patterns::Integer(0),
                        "Initial global refinement level");

      prm.declare_entry("Maximum cycles", "10",
                        Patterns::Integer(0),
                        "Maximum number of h-refinement cycles");

      prm.declare_entry("Maximum h-level", "6",
                        Patterns::Integer(0,20),
                        "Number of h-refinement levels in the discretisation");

      prm.declare_entry("Refinement fraction", "0.3",
                        Patterns::Double(0.0,1.0),
                        "Fraction of cells to refine");

      prm.declare_entry("Coarsening fraction", "0.03",
                        Patterns::Double(0.0,1.0),
                        "Fraction of cells to coarsen");

      prm.declare_entry("Force manifold_refinement", "false",
                        Patterns::Bool(),
                        "Force adaptive refinement at manifolds");
    }
    prm.leave_subsection();
  }

  void Refinement::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Refinement");
    {
      refinement_strategy = prm.get("Refinement strategy");
      n_global_refinements = prm.get_integer("Initial global refinements");
      n_cycles_max = prm.get_integer("Maximum cycles");
      n_levels_max = prm.get_integer("Maximum h-level");
      frac_refine = prm.get_double("Refinement fraction");
      frac_coarsen = prm.get_double("Coarsening fraction");
      force_manifold_refinement = prm.get_bool("Force manifold_refinement");
    }
    prm.leave_subsection();
  }

// @sect4{Materials}

// We also need the shear modulus $ \mu $ and Poisson ration $ \nu $ for the
// neo-Hookean material. We will let each benchmark problem set the shear modulus,
// but will leave the level of incompressibility a flexible quantity.
  struct Materials
  {
    double mu_r_air;
    double mu_r_membrane;
    double mu;
    double nu;
    double free_space_mu;
    double free_space_nu;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void Materials::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material properties");
    {
      prm.declare_entry("Membrane relative permeability", "1.0",
                        Patterns::Double(1e-9),
                        "Relative permeability of the toroidal membrane");
      prm.declare_entry("Shear modulus", "0.03",
                        Patterns::Double(),
                        "Shear modulus (Lame 2nd parameter)");
      prm.declare_entry("Poisson's ratio", "0.45",
                        Patterns::Double(-1.0,0.5),
                        "Poisson's ratio");
      prm.declare_entry("Free space shear modulus", "0.0003",
                        Patterns::Double(),
                        "Shear modulus for free space material");
      prm.declare_entry("Free space poisson's ratio", "0.45",
                        Patterns::Double(-1.0,0.5),
                        "Poisson's ratio for free space material");
    }
    prm.leave_subsection();
  }

  void Materials::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material properties");
    {
      mu_r_air = 1.0;
      mu_r_membrane = prm.get_double("Membrane relative permeability");
      mu = prm.get_double("Shear modulus");
      nu = prm.get_double("Poisson's ratio");
      free_space_mu = prm.get_double("Free space shear modulus");
      free_space_nu = prm.get_double("Free space poisson's ratio");
    }
    prm.leave_subsection();
  }

// @sect4{Linear solver}

// Next, we choose both solver and preconditioner settings.  The use of an
// effective preconditioner is critical to ensure convergence when a large
// nonlinear motion occurs within a Newton increment.
  struct LinearSolver
  {
    std::string lin_slvr_type;
    double      lin_slvr_tol;
    double      lin_slvr_max_it;
    std::string preconditioner_type;
    double      preconditioner_relaxation;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void LinearSolver::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear solver");
    {
      prm.declare_entry("Solver type", "Iterative",
                        Patterns::Selection("Iterative|Direct"),
                        "Type of solver used to solve the linear system");

      prm.declare_entry("Residual", "1e-6",
                        Patterns::Double(0.0),
                        "Linear solver residual (scaled by residual norm)");

      prm.declare_entry("Max iteration multiplier", "1",
                        Patterns::Double(0.0),
                        "Linear solver iterations (multiples of the system matrix size)");

      prm.declare_entry("Preconditioner type", "ssor",
                        Patterns::Selection("jacobi|ssor|AMG"),
                        "Type of preconditioner");

      prm.declare_entry("Preconditioner relaxation", "0.65",
                        Patterns::Double(0.0),
                        "Preconditioner relaxation value");
    }
    prm.leave_subsection();
  }

  void LinearSolver::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear solver");
    {
      lin_slvr_type = prm.get("Solver type");
      lin_slvr_tol = prm.get_double("Residual");
      lin_slvr_max_it = prm.get_double("Max iteration multiplier");
      preconditioner_type = prm.get("Preconditioner type");
      preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
    }
    prm.leave_subsection();
  }

  // Nonlinear solver parameters like max. Newton-Raphson iterations and tolerances for RHS and solution
  struct NonlinearSolver
  {
      unsigned int max_iterations_NR;
      double tol_f;
      double tol_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
  };

  void NonlinearSolver::declare_parameters(ParameterHandler &prm)
  {
      prm.enter_subsection("Nonlinear solver");
      {
          prm.declare_entry("Max iterations Newton-Raphson", "10",
                            Patterns::Integer(0),
                            "Number of Newton-Raphson iterations allowed");

          prm.declare_entry("Tolerance force", "1.0e-9",
                            Patterns::Double(0.0),
                            "Force residual tolerance");

          prm.declare_entry("Tolerance displacement", "1.0e-6",
                            Patterns::Double(0.0),
                            "Displacement error tolerance");
      }
      prm.leave_subsection();
  }

  void NonlinearSolver::parse_parameters(ParameterHandler &prm)
  {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f = prm.get_double("Tolerance force");
        tol_u = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
  }

// @sect4{All parameters}

// Finally we consolidate all of the above structures into a single container
// that holds all of our run-time selections.
  struct AllParameters :
    public LoadStep,
    public ProblemType,
    public BoundaryConditions,
    public FESystem,
    public Geometry,
    public Refinement,
    public Materials,
    public LinearSolver,
    public NonlinearSolver

  {
    AllParameters(const std::string &input_file);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  AllParameters::AllParameters(const std::string &input_file)
  {
    ParameterHandler prm;
    declare_parameters(prm);
    const bool print_default_prm_file = true;
    try
    {
        prm.parse_input(input_file);
    }
    catch (const PathSearch::ExcFileNotFound &)
    {
        std::cerr << "ParameterHandler::parse_input: could not open file <"
                  << input_file
                  << "> for reading."
                  << std::endl;
        if (print_default_prm_file)
          {
            std::cerr << "Trying to make file <"
                      << input_file
                      << "> with default values for you."
                      << std::endl;
            std::ofstream output (input_file);
            prm.print_parameters (output,
                                  ParameterHandler::OutputStyle::Text);
          }
    }
    parse_parameters(prm);
  }

  void AllParameters::declare_parameters(ParameterHandler &prm)
  {
    LoadStep::declare_parameters(prm);
    ProblemType::declare_parameters(prm);
    BoundaryConditions::declare_parameters(prm);
    FESystem::declare_parameters(prm);
    Geometry::declare_parameters(prm);
    Refinement::declare_parameters(prm);
    Materials::declare_parameters(prm);
    LinearSolver::declare_parameters(prm);
    NonlinearSolver::declare_parameters(prm);
  }

  void AllParameters::parse_parameters(ParameterHandler &prm)
  {
    LoadStep::parse_parameters(prm);
    ProblemType::parse_parameters(prm);
    BoundaryConditions::parse_parameters(prm);
    FESystem::parse_parameters(prm);
    Geometry::parse_parameters(prm);
    Refinement::parse_parameters(prm);
    Materials::parse_parameters(prm);
    LinearSolver::parse_parameters(prm);
    NonlinearSolver::parse_parameters(prm);
  }
}

// @sect3{Nonconstant coefficients}

template <int dim>
class Geometry
{
public:
  Geometry (const double &torus_major_radius,
            const double &torus_minor_radius_inner,
            const double &torus_minor_radius_outer)
    : torus_major_radius(torus_major_radius),
      torus_minor_radius_inner(torus_minor_radius_inner),
      torus_minor_radius_outer(torus_minor_radius_outer)
  {
    torus_membrane_circ_axis[0] = torus_major_radius;
  }

  virtual ~Geometry () {}

  bool on_radius_outer (const Point<dim> &p) const
  {
    const double dist_from_axis = torus_membrane_circ_axis.distance(p);
    return std::abs(dist_from_axis - torus_minor_radius_outer) < 1e-9;
  }

  bool on_radius_inner (const Point<dim> &p) const
  {
    const double dist_from_axis = torus_membrane_circ_axis.distance(p);
    return std::abs(dist_from_axis - torus_minor_radius_inner) < 1e-9;
  }

  bool within_membrane (const Point<dim> &p) const
  {
    const double dist_from_axis = torus_membrane_circ_axis.distance(p);
    return (dist_from_axis >= torus_minor_radius_inner) && (dist_from_axis <= torus_minor_radius_outer);
  }

  const Point<dim> &
  get_membrane_minor_radius_centre () const
  {
    return torus_membrane_circ_axis;
  }

  const double &
  get_torus_minor_radius_outer () const
  {
    return torus_minor_radius_outer;
  }

private:

  const double torus_major_radius;
  const double torus_minor_radius_inner;
  const double torus_minor_radius_outer;

  Point<dim> torus_membrane_circ_axis;
};

template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient (const Geometry<dim> &geometry,
               const double        &mu_r_air,
               const double        &mu_r_membrane)
    : Function<dim>(),
      geometry (geometry),
      mu_0 (4*M_PI*1e-7),
      mu_air(mu_r_air*mu_0),
      mu_membrane(mu_r_membrane*mu_0)
  { }

  virtual ~Coefficient () {}

  virtual double
  value (const Point<dim> &p,
         const unsigned int component = 0) const;

  virtual void
  value_list (const std::vector<Point<dim> > &points,
              std::vector<double>            &values,
              const unsigned int              component = 0) const;

  const Geometry<dim> &geometry;

  const double mu_0;
  const double mu_air;
  const double mu_membrane;
};



template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
  if (geometry.within_membrane(p) == true)
    return mu_membrane;
  else
    return mu_air;
}



template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points,
          ExcDimensionMismatch (values.size(), n_points));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  for (unsigned int i=0; i<n_points; ++i)
    {
      if (geometry.within_membrane(points[i]) == true)
        values[i] = mu_membrane;
      else
        values[i] = mu_membrane;
    }
}


template <int dim>
class LinearScalarPotential : public Function<dim>
{
public:
  LinearScalarPotential (const double potential_difference_per_unit_length)
    : potential_difference_per_unit_length (potential_difference_per_unit_length)
  { }

  virtual ~LinearScalarPotential () {}

  virtual double
  value (const Point<dim> &p,
         const unsigned int component = 0) const;

  virtual void
  value_list (const std::vector<Point<dim> > &points,
              std::vector<double>            &values,
              const unsigned int              component = 0) const;

  const double potential_difference_per_unit_length;
};



template <int dim>
double LinearScalarPotential<dim>::value (const Point<dim> &p,
                                          const unsigned int) const
{
  return -p[1]*potential_difference_per_unit_length;
}



template <int dim>
void LinearScalarPotential<dim>::value_list (const std::vector<Point<dim> > &points,
                                             std::vector<double>            &values,
                                             const unsigned int              component) const
{
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points,
          ExcDimensionMismatch (values.size(), n_points));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],component);
}


template <int dim>
class RefinementStrategy
{
public:
  RefinementStrategy (const std::string &refinement_strategy)
    : _use_h_refinement (refinement_strategy == "h-GMR" ||
                         refinement_strategy == "h-AMR" ||
                         refinement_strategy == "hp-AMR"),
      _use_p_refinement (refinement_strategy == "p-GMR" ||
                         refinement_strategy == "p-AMR" ||
                         refinement_strategy == "hp-AMR"),
      _use_AMR (refinement_strategy == "h-AMR" ||
                refinement_strategy == "p-AMR")
  {}

  bool use_h_refinement (void) const
  {
    return _use_h_refinement;
  }
  bool use_p_refinement (void) const
  {
    return _use_p_refinement;
  }
  bool use_hp_refinement (void) const
  {
    return use_h_refinement() & use_p_refinement();
  }
  bool use_AMR (void) const
  {
    return _use_AMR;
  }
  bool use_GR (void) const
  {
    return !use_AMR();
  }

private:
  const bool _use_h_refinement;
  const bool _use_p_refinement;
  const bool _use_AMR;
};

// Neo-Hookean nonlinear constitutive material model
/*
 * Used strain energy function here:
 * Psi(C) = Psi = mu * [C : I - I : I - 2 * ln(J)] / 2 + lambda * (ln(J))^2 / 2     // Elasticity contribution
 *                - mu_0 * mu_r * [J * C_inv : outer_product(H, H)]                 // Magnetic contribution
 *
 * Parameters: mu: shear modulus,
 *             mu_0: free space permeability
 *             mu_r: relative permeability of the magneto-elastic membrane material
 *             C: right Cauchy-Green deformation tensor
 *             I: second order Identity tensor
 *             J: Jacobian := det(F) with F: Deformation gradient
 *             lambda: Lame 1st parameter
 *             H: appied magnetic field
 * */
template <int dim, int dim_Tensor>
class Material_Neo_Hookean_Two_Field
{
public:
    Material_Neo_Hookean_Two_Field(const double mu, // Lame 2nd parameter: shear modulus
                                   const double nu) // Poisson ratio
        :
          mu_(mu),
          nu_(nu),
          kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu))),
          c_1(0.5 * mu),
          lambda(kappa - ((2.0 * mu) / 3.0)),
          det_F(1.0),
          phi(0.0)
    {
        Assert(kappa > 0.0, ExcInternalError());
    }

    ~Material_Neo_Hookean_Two_Field(){}

    void update_material_data(const Tensor<2, dim_Tensor> &F,
                              const double phi_in)
    {
        det_F = determinant(F);
//        std::cout << "det F: " << det_F << std::endl;
        phi = phi_in;

        Assert(det_F > 0.0, ExcInternalError());
    }

    double get_det_F() const
    {
        return det_F;
    }

    // Get 2nd Piola-Kirchoff stress tensor
    SymmetricTensor<2, dim_Tensor> get_2nd_Piola_Kirchoff_stress(const Tensor<2, dim_Tensor> &F) const
    {
        const SymmetricTensor<2, dim_Tensor> C = Physics::Elasticity::Kinematics::C(F);
        const SymmetricTensor<2, dim_Tensor> C_inv = invert(C);
        return ( (mu_ * Physics::Elasticity::StandardTensors<dim_Tensor>::I) -
                ( (mu_ - lambda * std::log(det_F) ) * C_inv ));
    }

    // Get the 4th order material elasticity tensor
    SymmetricTensor<4, dim_Tensor> get_4th_order_material_elasticity(const Tensor<2, dim_Tensor> &F) const
    {
        const SymmetricTensor<2, dim_Tensor> C = Physics::Elasticity::Kinematics::C(F);
        const SymmetricTensor<2, dim_Tensor> C_inv = invert(C);
        const SymmetricTensor<4, dim_Tensor> C_inv_C_inv = outer_product(C_inv, C_inv);

        return ( (lambda * C_inv_C_inv) +
                 ( (2.0 * lambda * std::log(det_F) - 2.0 * mu_ ) *
                 (Physics::Elasticity::StandardTensors<dim_Tensor>::dC_inv_dC(F)) ) );

    }

    double get_phi() const
    {
        return phi;
    }

private:
    const double mu_; // shear modulus
    const double nu_; // Poisson ratio
    const double kappa; // bulk modulus
    const double c_1; // material constant
    const double lambda; // Lame 1st parameter
    double det_F;
    double phi; // scalar magnetic potential
};

// Quadrature point history class
template <int dim, int dim_Tensor>
class PointHistory
{
  public:
    PointHistory()
        :
          F_inv(Physics::Elasticity::StandardTensors<dim_Tensor>::I),
          second_Piola_Kirchoff_stress(SymmetricTensor<2, dim_Tensor>()),
          fourth_order_material_elasticity(SymmetricTensor<4, dim_Tensor>())
    {}

    virtual ~PointHistory(){}

    void setup_lqp (const double mu, const double nu)
    {
//        Assert(!material, ExcInternalError());
        material = std::make_shared<Material_Neo_Hookean_Two_Field<dim, dim_Tensor> >(mu,nu);
        Assert(material, ExcInternalError());
        update_values(Tensor<2, dim_Tensor>(), 0.0);
    }

    void update_values(const Tensor<2, dim_Tensor> &Grad_u_n,
                       const double phi)
    {
        const Tensor<2, dim_Tensor> F = Physics::Elasticity::Kinematics::F(Grad_u_n);
        F_inv = invert(F);
        Assert(material, ExcInternalError());
        material->update_material_data(F, phi);
        second_Piola_Kirchoff_stress = material->get_2nd_Piola_Kirchoff_stress(F);
        fourth_order_material_elasticity = material->get_4th_order_material_elasticity(F);
//        std::cout << "F: " << F << "\n" << "S: " << second_Piola_Kirchoff_stress << std::endl;
    }

    double get_det_F() const
    {
        return material->get_det_F();
    }

    const Tensor<2, dim_Tensor> &get_F_inv() const
    {
        return F_inv;
    }

    const SymmetricTensor<2, dim_Tensor> &get_second_Piola_Kirchoff_stress() const
    {
        return second_Piola_Kirchoff_stress;
    }

    const SymmetricTensor<4, dim_Tensor> &get_4th_order_material_elasticity() const
    {
        return fourth_order_material_elasticity;
    }

    double get_phi() const
    {
        return material->get_phi();
    }

private:
    std::shared_ptr<Material_Neo_Hookean_Two_Field<dim,dim_Tensor> > material;
    Tensor<2, dim_Tensor> F_inv;
    SymmetricTensor<2, dim_Tensor> second_Piola_Kirchoff_stress;
    SymmetricTensor<4, dim_Tensor> fourth_order_material_elasticity;
};

// Class to store load step data
// For now we assume a constant load increment step size
class LoadStep
{
public:
    LoadStep(const double total_load,
             const double delta_load)
        :
          loadstep(0),
          current_load(0.0),
          total_load(total_load),
          delta_load(delta_load)
    {}

    virtual ~LoadStep() {}

    double current() const
    {
        return current_load;
    }

    double final() const
    {
        return total_load;
    }

    double get_delta_load() const
    {
        return delta_load;
    }

    unsigned int get_loadstep() const
    {
        return loadstep;
    }

    void increment()
    {
        current_load += delta_load;
        ++loadstep;
    }

    void reset()
    {
        loadstep = 0;
        current_load = 0.0;
    }

private:
    unsigned int loadstep;
    double current_load;
    const double total_load;
    const double delta_load;
};

// @sect3{The <code>MSP_Toroidal_Membrane</code> class template}

template <int dim>
class MSP_Toroidal_Membrane
{
public:
  MSP_Toroidal_Membrane (const std::string &input_file);
  ~MSP_Toroidal_Membrane ();

  void run ();

private:
  void set_initial_fe_indices ();
  void setup_system ();
  void setup_quadrature_point_history();
  void update_qph_incremental(const TrilinosWrappers::MPI::BlockVector &solution_delta);
  void make_constraints (ConstraintMatrix &constraints, const int &itr_nr);
  void assemble_system ();
  void solve (TrilinosWrappers::MPI::BlockVector &newton_update);
  void solve_nonlinear_system(TrilinosWrappers::MPI::BlockVector &solution_delta);
  void make_grid ();
  void make_grid_manifold_ids ();
  void refine_grid ();
  void compute_error();
  void output_results (const unsigned int cycle,
                       const unsigned int load_step_number) const;
  void postprocess_energy ();
  TrilinosWrappers::MPI::BlockVector
  get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta) const;
  void print_convergence_header();
  void print_convergence_footer();

  MPI_Comm           mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  mutable ConditionalOStream pcout;
  mutable TimerOutput computing_timer;

  Parameters::AllParameters parameters;

  // irrespective of 3D or axisymmetric problem dim_Tensor value will always be 3
  static const int dim_Tensor = 3; // dimensions for the kinematics and kinetics tensors
  CellDataStorage<typename Triangulation<dim>::cell_iterator,
                  PointHistory<dim, dim_Tensor> > quadrature_point_history;

  const Geometry<dim>      geometry;
  const types::manifold_id manifold_id_inner_radius;
  const types::manifold_id manifold_id_outer_radius;
  const types::manifold_id manifold_id_magnet;
  SphericalManifold<dim>   manifold_inner_radius;
  SphericalManifold<dim>   manifold_outer_radius;
  TransfiniteInterpolationManifold<dim>   manifold_magnet;
  const types::boundary_id boundary_id_magnet;
  const types::material_id material_id_toroid;
  const types::material_id material_id_vacuum;
  const types::material_id material_id_bar_magnet;

  parallel::shared::Triangulation<dim>      triangulation;
  RefinementStrategy<dim> refinement_strategy;

  std::vector<unsigned int>  degree_collection;
  hp::FECollection<dim>      fe_collection;
  hp::MappingCollection<dim> mapping_collection;
  hp::DoFHandler<dim>        hp_dof_handler;
  hp::QCollection<dim>       qf_collection_cell;
  hp::QCollection<dim-1>     qf_collection_face;

  std::vector<IndexSet> all_locally_owned_dofs;
  IndexSet              locally_owned_dofs;
  IndexSet              locally_relevant_dofs;
  ConstraintMatrix      constraints;
  ConstraintMatrix      hanging_node_constraints;

  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::MPI::BlockVector  system_rhs;
  TrilinosWrappers::MPI::BlockVector  solution;

  Vector<float>        estimated_error_per_cell; // For Kelly error estimator

  Coefficient<dim>     function_material_coefficients;

  mutable ConvergenceTable     convergence_table;  

  std::vector<IndexSet> locally_owned_partitioning, locally_relevant_partitioning;
  static const unsigned int n_blocks = 2;
  static const unsigned int phi_component = 0;
  static const unsigned int first_u_component = 1;
  static const unsigned int n_components = dim + 1;
  const FEValuesExtractors::Scalar phi_fe;
  const FEValuesExtractors::Vector u_fe;

  enum
  {
      phi_block = 0,
      u_block = 1
  };

  enum
  {
      potential_component = 0,
      displacement_r_component = 1,
      displacement_z_component = 2,
      displacement_theta_component = 3
  };

  std::vector<types::global_dof_index> dofs_per_block;

  // A data structure to hold a number of variables to store norms and update norms
  struct Errors
  {
      Errors()
          :
            norm(1.0), u(1.0), phi(1.0)
      {}
      void reset()
      {
          norm = 1.0;
          u = 1.0;
          phi = 1.0;
      }

      void normalize(const Errors &rhs)
      {
          if (rhs.norm != 0.0)
              norm /= rhs.norm;
          if (rhs.u != 0.0)
              u /= rhs.u;
          if (rhs.phi != 0.0)
              phi /= rhs.phi;
      }

      double norm, u, phi;
  };

  Errors error_residual, error_residual_0, error_residual_norm,
         error_update, error_update_0, error_update_norm;

  // Functions to calculate the errors
  void
  get_error_residual(Errors &error_residual);

  void
  get_error_update(const TrilinosWrappers::MPI::BlockVector &newton_update,
                   Errors &error_update);

  // Inputs: Total load at the end of load steps &
  //         Load increments in each step
  LoadStep loadstep;
};



#endif // MSP_TOROIDAL_MEMBRANE_H
