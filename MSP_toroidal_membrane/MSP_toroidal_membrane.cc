#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

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

#include <deal.II/distributed/shared_tria.h>

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
// @sect4{Boundary conditions}

  struct BoundaryConditions
  {
    double potential_difference_per_unit_length;

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

    }
    prm.leave_subsection();
  }

  void BoundaryConditions::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Boundary conditions");
    {
      potential_difference_per_unit_length = prm.get_double("Potential difference per unit length");
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

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
  };

  void Geometry::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      prm.declare_entry("Mesh file", "../mesh/toroidal_membrane.inp",
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
    }
    prm.leave_subsection();
  }

  void Materials::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material properties");
    {
      mu_r_air = 1.0;
      mu_r_membrane = prm.get_double("Membrane relative permeability");
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

// @sect4{All parameters}

// Finally we consolidate all of the above structures into a single container
// that holds all of our run-time selections.
  struct AllParameters :
    public BoundaryConditions,
    public FESystem,
    public Geometry,
    public Refinement,
    public Materials,
    public LinearSolver

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
    prm.parse_input(input_file);
    parse_parameters(prm);
  }

  void AllParameters::declare_parameters(ParameterHandler &prm)
  {
    BoundaryConditions::declare_parameters(prm);
    FESystem::declare_parameters(prm);
    Geometry::declare_parameters(prm);
    Refinement::declare_parameters(prm);
    Materials::declare_parameters(prm);
    LinearSolver::declare_parameters(prm);
  }

  void AllParameters::parse_parameters(ParameterHandler &prm)
  {
    BoundaryConditions::parse_parameters(prm);
    FESystem::parse_parameters(prm);
    Geometry::parse_parameters(prm);
    Refinement::parse_parameters(prm);
    Materials::parse_parameters(prm);
    LinearSolver::parse_parameters(prm);
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
  void make_constraints (ConstraintMatrix &constraints);
  void assemble_system ();
  void solve ();
  void make_grid ();
  void make_grid_manifold_ids ();
  void refine_grid ();
  void compute_error();
  void output_results (const unsigned int cycle) const;

  MPI_Comm           mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  mutable ConditionalOStream pcout;
  mutable TimerOutput computing_timer;

  Parameters::AllParameters parameters;

  const Geometry<dim>      geometry;
  const types::manifold_id manifold_id_inner_radius;
  const types::manifold_id manifold_id_outer_radius;
  SphericalManifold<dim>   manifold_inner_radius;
  SphericalManifold<dim>   manifold_outer_radius;
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

  std::vector<IndexSet> all_locally_owned_dofs;
  IndexSet              locally_owned_dofs;
  IndexSet              locally_relevant_dofs;
  ConstraintMatrix      constraints;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector  system_rhs;
  TrilinosWrappers::MPI::Vector  solution;

  Vector<float>        estimated_error_per_cell; // For Kelly error estimator

  Coefficient<dim>     function_material_coefficients;

  mutable ConvergenceTable     convergence_table;
};


// @sect3{The <code>MSP_Toroidal_Membrane</code> class implementation}

// @sect4{MSP_Toroidal_Membrane::MSP_Toroidal_Membrane}

template <int dim>
MSP_Toroidal_Membrane<dim>::MSP_Toroidal_Membrane (const std::string &input_file)
  :
  mpi_communicator(MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout(std::cout, this_mpi_process == 0),
  computing_timer(mpi_communicator,
                 pcout,
                 TimerOutput::never,
                 TimerOutput::wall_times),
  parameters (input_file),
  geometry (parameters.torus_major_radius*parameters.grid_scale,
           parameters.torus_minor_radius_inner*parameters.grid_scale,
           parameters.torus_minor_radius_outer*parameters.grid_scale),
  manifold_id_inner_radius (100),
  manifold_id_outer_radius (101),
  manifold_inner_radius(geometry.get_membrane_minor_radius_centre()),
  manifold_outer_radius(geometry.get_membrane_minor_radius_centre()),
  boundary_id_magnet (10),
  material_id_toroid(1),
  material_id_vacuum(2),
  material_id_bar_magnet(3),
  triangulation(mpi_communicator,
                Triangulation<dim>::maximum_smoothing),
  refinement_strategy (parameters.refinement_strategy),
  hp_dof_handler (triangulation),
  function_material_coefficients (geometry,
                                 parameters.mu_r_air,
                                 parameters.mu_r_membrane)
{
  AssertThrow(parameters.poly_degree_max >= parameters.poly_degree_min, ExcInternalError());

  for (unsigned int degree = parameters.poly_degree_min;
       degree <= parameters.poly_degree_max; ++degree)
    {
      degree_collection.push_back(degree); // Polynomial degree
      fe_collection.push_back(FE_Q<dim>(degree));
      mapping_collection.push_back(MappingQGeneric<dim>(degree));
      qf_collection_cell.push_back(QGauss<dim>  (degree + 1));
    }
}


// @sect4{MSP_Toroidal_Membrane::~MSP_Toroidal_Membrane}

template <int dim>
MSP_Toroidal_Membrane<dim>::~MSP_Toroidal_Membrane ()
{
  hp_dof_handler.clear ();
}


// @sect4{MSP_Toroidal_Membrane::setup_system}

template <int dim>
void MSP_Toroidal_Membrane<dim>::set_initial_fe_indices()
{
  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = hp_dof_handler.begin_active(),
  endc = hp_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
    if (cell->is_locally_owned() == false) continue;
//      if (cell->subdomain_id() != this_mpi_process) continue;

      if (geometry.within_membrane(cell->center()))
        cell->set_active_fe_index(0); // 1 for p-refinement test
      else
        cell->set_active_fe_index(0);
    }
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::make_constraints (ConstraintMatrix &constraints)
{
  // Assume a homogeneous magnetic field is generated at
  // an infinite distance from the particle
//  VectorTools::interpolate_boundary_values (hp_dof_handler,
//                                            boundary_id_magnet,
//                                            LinearScalarPotential<dim>(parameters.potential_difference_per_unit_length),
//                                            constraints);

    std::map< types::global_dof_index, Point<dim> > support_points;
    DoFTools::map_dofs_to_support_points(mapping_collection, hp_dof_handler, support_points);
    LinearScalarPotential<dim> linear_scalar_potential(parameters.potential_difference_per_unit_length);

    for(auto it : support_points)
    {
        const auto dof_index = it.first;
        const auto supp_point = it.second;

        // Check for the support point if inside the permanent magnet region:
        // In 2D axisymmetric we have x,z <=> r,z so need to compare 0th and 1st component of point
        // In 3D we have x,z,y <=> r,z,theta so need to compare 0th and 1st component of point
        if( std::abs(supp_point[0]) < parameters.bounding_box_r && // X coord of support point less than magnet radius...
            std::abs(supp_point[1]) < parameters.bounding_box_z) // Z coord of support point less than magnet height
        {
//            pcout << "DoF index: " << dof_index << "    " << "point: " << supp_point << std::endl;
            const double potential_value = linear_scalar_potential.value(supp_point);
//            pcout << "Potential value: " << potential_value << std::endl;
            constraints.add_line(dof_index);
            constraints.set_inhomogeneity(dof_index, potential_value);
        }
    }
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::setup_system ()
{
  {
    TimerOutput::Scope timer_scope (computing_timer, "Setup: distribute DoFs");

    // Partition triangulation if using Triangulation<dim>
//    GridTools::partition_triangulation (n_mpi_processes,
//                                        triangulation);

    // Distribute DoFs
    hp_dof_handler.distribute_dofs (fe_collection);
   // When using parallel::shared::Triangulation no need to do this
//    DoFRenumbering::subdomain_wise (hp_dof_handler);

    locally_owned_dofs.clear();
    locally_relevant_dofs.clear();
    all_locally_owned_dofs = DoFTools::locally_owned_dofs_per_subdomain (hp_dof_handler);
    locally_owned_dofs = all_locally_owned_dofs[this_mpi_process];
//    DoFTools::extract_locally_relevant_dofs (hp_dof_handler, locally_relevant_dofs); // Old method
    locally_relevant_dofs = DoFTools::locally_relevant_dofs_per_subdomain (hp_dof_handler)[this_mpi_process];
  }

  {
    TimerOutput::Scope timer_scope (computing_timer, "Setup: constraints");

    constraints.clear ();
//    constraints.reinit(locally_relevant_dofs); // make_hanging_node_constraints needs locally owned and ghost cells. Cannot do it with normal triangulation with subdomain is indicator of locally owned cells
    DoFTools::make_hanging_node_constraints (hp_dof_handler,
                                             constraints);
//    make_dirichlet_constraints(constraints);
    make_constraints(constraints);
    constraints.close ();
  }

  {
    TimerOutput::Scope timer_scope (computing_timer, "Setup: matrix, vectors");

    std::vector<dealii::types::global_dof_index> n_locally_owned_dofs_per_processor (n_mpi_processes);
    {
      AssertThrow(all_locally_owned_dofs.size() == n_locally_owned_dofs_per_processor.size(), ExcInternalError());
      for (unsigned int i=0; i < n_locally_owned_dofs_per_processor.size(); ++i)
        n_locally_owned_dofs_per_processor[i] = all_locally_owned_dofs[i].n_elements();
    }

    TrilinosWrappers::SparsityPattern sp (locally_owned_dofs,
                                          locally_owned_dofs,
                                          locally_relevant_dofs,
                                          mpi_communicator);
    DoFTools::make_sparsity_pattern (hp_dof_handler,
                                     sp,
                                     constraints,
                                     /* keep constrained dofs */ false,
                                     Utilities::MPI::this_mpi_process(mpi_communicator)); //keep constrained dofs as we need undocndensed matrices as well

    sp.compress();
    /*
     * dealii::SparsityTools::distribute_sparsity_pattern (sp,
                                                        n_locally_owned_dofs_per_processor,
                                                        mpi_communicator,
                                                        locally_relevant_dofs);
                                                        */
    system_matrix.reinit (sp);
    system_rhs.reinit(locally_owned_dofs,
                      mpi_communicator);
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);
  }
}


// @sect4{MSP_Toroidal_Membrane::assemble_system}

template <int dim>
void MSP_Toroidal_Membrane<dim>::assemble_system ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Assembly");

  hp::FEValues<dim> hp_fe_values (mapping_collection,
                                  fe_collection,
                                  qf_collection_cell,
                                  update_values |
                                  update_gradients |
                                  update_quadrature_points |
                                  update_JxW_values);

  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = hp_dof_handler.begin_active(),
  endc = hp_dof_handler.end();
  for (; cell!=endc; ++cell)
    {
    if (cell->is_locally_owned() == false) continue;
//      if (cell->subdomain_id() != this_mpi_process) continue;

      hp_fe_values.reinit(cell);
      const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
      const unsigned int  &n_q_points = fe_values.n_quadrature_points;
      const unsigned int  &n_dofs_per_cell = fe_values.dofs_per_cell;
      const std::vector<Point<dim> > &quadrature_points = fe_values.get_quadrature_points();

      FullMatrix<double>   cell_matrix (n_dofs_per_cell, n_dofs_per_cell);
      Vector<double>       cell_rhs (n_dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (n_dofs_per_cell);
      std::vector<double>    coefficient_values (n_q_points);

      function_material_coefficients.value_list (fe_values.get_quadrature_points(),
                                                 coefficient_values);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          const double mu_r_mu_0 = coefficient_values[q_index];
          // Get the x co-ord to the quadrature point
          const double radial_distance = quadrature_points[q_index][0];
          // If dim == 2, assembly using axisymmetric formulation
          const double coord_transformation_scaling = ( dim == 2
                                                        ?
                                                          2.0 * dealii::numbers::PI * radial_distance
                                                        :
                                                          1.0);

          for (unsigned int i=0; i<n_dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<=i; ++j)
                cell_matrix(i,j) += fe_values.shape_grad(i,q_index) *
                                    mu_r_mu_0*
                                    coord_transformation_scaling *
                                    fe_values.shape_grad(j,q_index) *
                                    fe_values.JxW(q_index);
            }
        }

      // Finally, we need to copy the lower half of the local matrix into the
      // upper half:
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < n_dofs_per_cell; ++j)
          cell_matrix(i, j) = cell_matrix(j, i);

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


// @sect4{MSP_Toroidal_Membrane::solve}

template <int dim>
void MSP_Toroidal_Membrane<dim>::solve ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Solve linear system");

  TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs,
                                                     mpi_communicator);
//  distributed_solution = solution;

  SolverControl solver_control (parameters.lin_slvr_max_it*system_matrix.m(),
                                parameters.lin_slvr_tol);
  if (parameters.lin_slvr_type == "Iterative")
    {

      TrilinosWrappers::SolverCG solver (solver_control);

      // Default settings for AMG preconditioner are
      // good for a Laplace problem
      std::unique_ptr<TrilinosWrappers::PreconditionBase> preconditioner;
      if (parameters.preconditioner_type == "jacobi")
        {
          TrilinosWrappers::PreconditionJacobi *ptr_prec
            = new TrilinosWrappers::PreconditionJacobi ();

          TrilinosWrappers::PreconditionJacobi::AdditionalData
          additional_data (parameters.preconditioner_relaxation);

          ptr_prec->initialize(system_matrix,
                               additional_data);
          preconditioner.reset(ptr_prec);
        }
      else if (parameters.preconditioner_type == "ssor")
        {
          TrilinosWrappers::PreconditionSSOR *ptr_prec
            = new TrilinosWrappers::PreconditionSSOR ();

          TrilinosWrappers::PreconditionSSOR::AdditionalData
          additional_data (parameters.preconditioner_relaxation);

          ptr_prec->initialize(system_matrix,
                               additional_data);
          preconditioner.reset(ptr_prec);
        }
      else // AMG
        {
          TrilinosWrappers::PreconditionAMG *ptr_prec
            = new TrilinosWrappers::PreconditionAMG ();

          TrilinosWrappers::PreconditionAMG::AdditionalData additional_data;

          typename hp::DoFHandler<dim>::active_cell_iterator
          cell = hp_dof_handler.begin_active(),
          endc = hp_dof_handler.end();
          for (; cell!=endc; ++cell)
            {
              //    if (cell->is_locally_owned() == false) continue;
              if (cell->subdomain_id() != this_mpi_process) continue;

              const unsigned int cell_fe_idx = cell->active_fe_index();
              const unsigned int cell_poly = cell_fe_idx + 1;
              if (cell_poly > 1)
                {
                  additional_data.higher_order_elements = true;
                  break;
                }
            }
          {
            const int hoe = additional_data.higher_order_elements;
            additional_data.higher_order_elements
              = Utilities::MPI::max(hoe, mpi_communicator);
          }
          ptr_prec->initialize(system_matrix,
                               additional_data);
          preconditioner.reset(ptr_prec);
        }

      solver.solve (system_matrix,
                    distributed_solution,
                    system_rhs,
                    *preconditioner);
    }
  else // Direct
    {
      TrilinosWrappers::SolverDirect solver (solver_control);
      solver.solve (system_matrix,
                    distributed_solution,
                    system_rhs);
    }

  constraints.distribute (distributed_solution);
  solution = distributed_solution;

  pcout
      << "   Iterations: " << solver_control.last_step()
      << "  Residual: " << solver_control.last_value()
      << std::endl;
}


// @sect4{MSP_Toroidal_Membrane::refine_grid}


template <int dim>
void MSP_Toroidal_Membrane<dim>::compute_error ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Compute error");
  pcout << "   Computing errors" << std::endl;

  hp::QCollection<dim-1> EE_qf_collection_face_QGauss;
  for (unsigned int degree = parameters.poly_degree_min;
       degree <= parameters.poly_degree_max; ++degree)
    {
      EE_qf_collection_face_QGauss.push_back(QGauss<dim-1> (degree + 2));
    }

  TrilinosWrappers::MPI::Vector distributed_solution(locally_owned_dofs,
                                                     mpi_communicator);
  distributed_solution = solution;
  const Vector<double> localised_solution (distributed_solution);

  // --- Kelly Error estimator ---
  estimated_error_per_cell.reinit(triangulation.n_active_cells());
  estimated_error_per_cell = 0.0;
  KellyErrorEstimator<dim>::estimate (hp_dof_handler,
                                      EE_qf_collection_face_QGauss,
                                      typename FunctionMap<dim>::type(),
                                      localised_solution,
                                      estimated_error_per_cell,
                                      ComponentMask());
//                                      ,
//                                      /*coefficients = */ 0,
//                                      /*n_threads = */ numbers::invalid_unsigned_int,
//                                      /*subdomain_id = */this_mpi_process);
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::refine_grid ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Grid refinement");

  // Global refinement
  if (refinement_strategy.use_GR() == true)
    {
      if (refinement_strategy.use_h_refinement() == true)
        {
          AssertThrow (triangulation.n_global_levels() < parameters.n_levels_max, ExcInternalError());
          triangulation.refine_global (1);
        }
      else // p-refinement
        {
          AssertThrow(false, ExcNotImplemented());
//      AssertThrow(poly_refinement_strategy.use_p_refinement() == true, ExcInternalError());
        }
    }
  else // Adaptive mesh refinement
    {
      // Mark cells for adaptive mesh refinement...
      GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                       estimated_error_per_cell,
                                                       parameters.frac_refine,
                                                       parameters.frac_coarsen);

      if (parameters.force_manifold_refinement)
        {
          for (typename Triangulation<dim>::active_cell_iterator
               cell = triangulation.begin_active();
               cell != triangulation.end(); ++cell)
            {
              for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                {
                  if (cell->face(face)->manifold_id() == manifold_id_inner_radius ||
                      cell->face(face)->manifold_id() == manifold_id_outer_radius)
                    {
                      cell->clear_coarsen_flag();
                      cell->set_refine_flag();
                      continue;
                    }
                }
            }
        }

      // Check that there are no violations on maximum cell level
      // If so, then remove the marking
//      if (triangulation.n_global_levels() > static_cast<int>(parameters.n_levels_max))
      if (triangulation.n_levels() >= parameters.n_levels_max)
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active(parameters.n_levels_max-1);
             cell != triangulation.end(); ++cell)
          {
            cell->clear_refine_flag();
          }

      if (refinement_strategy.use_h_refinement() == true &&
          refinement_strategy.use_p_refinement() == false) // h-refinement
        {
          triangulation.execute_coarsening_and_refinement ();
          make_grid_manifold_ids();
        }
      else if (refinement_strategy.use_h_refinement() == false &&
               refinement_strategy.use_p_refinement() == true) // p-refinement
        {
          typename hp::DoFHandler<dim>::active_cell_iterator
          cell = hp_dof_handler.begin_active(),
          endc = hp_dof_handler.end();
          for (; cell!=endc; ++cell)
            {
              //    if (cell->is_locally_owned() == false) continue;
              if (cell->subdomain_id() != this_mpi_process)
                {
                  // Clear flags on non-owned cell that would
                  // be cleared on the owner processor anyway...
                  cell->clear_refine_flag();
                  cell->clear_coarsen_flag();
                  continue;
                }

              const unsigned int cell_fe_idx = cell->active_fe_index();
              const unsigned int cell_poly = cell_fe_idx + 1;

              if (cell->refine_flag_set())
                {
                  if (cell_poly < parameters.poly_degree_max)
                    cell->set_active_fe_index(cell_fe_idx+1);
                  cell->clear_refine_flag();
                }

              if (cell->coarsen_flag_set())
                {
                  if (cell_poly > parameters.poly_degree_min)
                    cell->set_active_fe_index(cell_fe_idx-1);
                  cell->clear_coarsen_flag();
                }

              AssertThrow(!(cell->refine_flag_set()), ExcInternalError());
              AssertThrow(!(cell->coarsen_flag_set()), ExcInternalError());
            }
        }
      else // hp-refinement
        {
          AssertThrow(refinement_strategy.use_hp_refinement() == true, ExcInternalError());
          AssertThrow(false, ExcNotImplemented());
        }


    }
}


template <int dim>
class MagneticFieldPostprocessor : public DataPostprocessorVector<dim>
{
public:
  MagneticFieldPostprocessor (const unsigned int material_id)
    : DataPostprocessorVector<dim> ("magnetic_field_"+std::to_string(material_id), update_gradients),
      material_id(material_id)
  {}

  virtual ~MagneticFieldPostprocessor() {}

  virtual void
  evaluate_scalar_field (const DataPostprocessorInputs::Scalar<dim> &input_data,
                         std::vector<Vector<double> >               &computed_quantities) const
  {
    // ensure that there really are as many output slots
    // as there are points at which DataOut provides the
    // gradients:
    AssertDimension (input_data.solution_gradients.size(),
                     computed_quantities.size());
    // then loop over all of these inputs:
    for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
      {
        // ensure that each output slot has exactly 'dim'
        // components (as should be expected, given that we
        // want to create vector-valued outputs), and copy the
        // gradients of the solution at the evaluation points
        // into the output slots:
        AssertDimension (computed_quantities[p].size(), dim);
        for (unsigned int d=0; d<dim; ++d)
        {
            auto current_cell = input_data.template get_cell<hp::DoFHandler<dim> >();
            if(current_cell->material_id() == material_id)
                computed_quantities[p][d] = -input_data.solution_gradients[p][d];
            else
                computed_quantities[p][d] = 0;
        }
      }
  }

private:
  const unsigned int material_id;
};



// @sect4{MSP_Toroidal_Membrane::output_results}

template<int dim, class DH=DoFHandler<dim> >
class FilteredDataOut : public DataOut<dim, DH>
{
public:
  FilteredDataOut (const unsigned int subdomain_id)
    :
    subdomain_id (subdomain_id)
  {}

  virtual ~FilteredDataOut() {}

  virtual typename DataOut<dim, DH>::cell_iterator
  first_cell ()
  {
    typename DataOut<dim, DH>::active_cell_iterator
    cell = this->dofs->begin_active();
    while ((cell != this->dofs->end()) &&
           (cell->subdomain_id() != subdomain_id))
      ++cell;
    return cell;
  }

  virtual typename DataOut<dim, DH>::cell_iterator
  next_cell (const typename DataOut<dim, DH>::cell_iterator &old_cell)
  {
    if (old_cell != this->dofs->end())
      {
        const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
        return
          ++(FilteredIterator
             <typename DataOut<dim, DH>::active_cell_iterator>
             (predicate,old_cell));
      }
    else
      return old_cell;
  }

private:
  const unsigned int subdomain_id;
};

template <int dim>
void MSP_Toroidal_Membrane<dim>::output_results (const unsigned int cycle) const
{
  TimerOutput::Scope timer_scope (computing_timer, "Output results");
  pcout << "   Outputting results" << std::endl;

  MagneticFieldPostprocessor<dim> mag_field_bar_magnet(material_id_bar_magnet);
  MagneticFieldPostprocessor<dim> mag_field_toroid(material_id_toroid); // Material ID for Toroid tube as read in from Mesh file
  MagneticFieldPostprocessor<dim> mag_field_vacuum(material_id_vacuum); // Material ID for free space
  FilteredDataOut< dim,hp::DoFHandler<dim> > data_out (this_mpi_process);

  data_out.attach_dof_handler (hp_dof_handler);

  data_out.add_data_vector (solution, "solution");
  data_out.add_data_vector (estimated_error_per_cell, "estimated_error");
  data_out.add_data_vector (solution, mag_field_bar_magnet);
  data_out.add_data_vector (solution, mag_field_toroid);
  data_out.add_data_vector (solution, mag_field_vacuum);

  // --- Additional data ---
  // Material coefficients; polynomial order; material id
  Vector<double> material_coefficients;
  Vector<double> polynomial_order;
  Vector<double> material_id;
  material_coefficients.reinit(triangulation.n_active_cells());
  polynomial_order.reinit(triangulation.n_active_cells());
  material_id.reinit(triangulation.n_active_cells());
  {
    unsigned int c = 0;
    typename Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell, ++c)
      {
//      if (cell->is_locally_owned() == false) continue;
        if (cell->subdomain_id() != this_mpi_process) continue;

        material_coefficients(c) = function_material_coefficients.value(cell->center());
        material_id(c) = cell->material_id();
      }
  }
  data_out.add_data_vector(material_id, "material_id");

  unsigned int max_used_poly_degree = 1;
  {
    unsigned int c = 0;
    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = hp_dof_handler.begin_active(),
    endc = hp_dof_handler.end();
    for (; cell!=endc; ++cell, ++c)
      {
//      if (cell->is_locally_owned() == false) continue;
        if (cell->subdomain_id() != this_mpi_process) continue;

        polynomial_order(c) = degree_collection[cell->active_fe_index()];
        max_used_poly_degree = std::max(max_used_poly_degree, cell->active_fe_index()+1);
      }

    max_used_poly_degree = Utilities::MPI::max(max_used_poly_degree, mpi_communicator);
  }
  data_out.add_data_vector (material_coefficients, "material_coefficients");
  data_out.add_data_vector (polynomial_order, "polynomial_order");

  std::vector<types::subdomain_id> partition_int (triangulation.n_active_cells());
  GridTools::get_subdomain_association (triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(),
                                    partition_int.end());
  data_out.add_data_vector (partitioning, "partitioning");

  data_out.build_patches(mapping_collection[max_used_poly_degree-1], max_used_poly_degree);
//  data_out.build_patches(max_used_poly_degree);
//  data_out.build_patches(*std::max_element(degree_collection.begin(),
//                                           degree_collection.end()));

  // Write out main data file
  struct Filename
  {
    static std::string get_filename_vtu (unsigned int process,
                                         unsigned int cycle,
                                         const unsigned int n_digits = 4)
    {
      std::ostringstream filename_vtu;
      filename_vtu
          << "solution-"
          << (std::to_string(dim) + "d")
          << "."
          << Utilities::int_to_string (process, n_digits)
          << "."
          << Utilities::int_to_string(cycle, n_digits)
          << ".vtu";
      return filename_vtu.str();
    }

    static std::string get_filename_pvtu (unsigned int timestep,
                                          const unsigned int n_digits = 4)
    {
      std::ostringstream filename_vtu;
      filename_vtu
          << "solution-"
          << (std::to_string(dim) + "d")
          << "."
          << Utilities::int_to_string(timestep, n_digits)
          << ".pvtu";
      return filename_vtu.str();
    }

    static std::string get_filename_pvd (void)
    {
      std::ostringstream filename_vtu;
      filename_vtu
          << "solution-"
          << (std::to_string(dim) + "d")
          << ".pvd";
      return filename_vtu.str();
    }
  };

  const std::string filename_vtu = Filename::get_filename_vtu(this_mpi_process, cycle);
  std::ofstream output(filename_vtu.c_str());
  data_out.write_vtu(output);

  // Collection of files written in parallel
  // This next set of steps should only be performed
  // by master process
  if (this_mpi_process == 0)
    {
      // List of all files written out at this timestep by all processors
      std::vector<std::string> parallel_filenames_vtu;
      for (unsigned int p=0; p < n_mpi_processes; ++p)
        {
          parallel_filenames_vtu.push_back(Filename::get_filename_vtu(p, cycle));
        }

      const std::string filename_pvtu (Filename::get_filename_pvtu(cycle));
      std::ofstream pvtu_master(filename_pvtu.c_str());
      data_out.write_pvtu_record(pvtu_master,
                                 parallel_filenames_vtu);

      // Time dependent data master file
      static std::vector<std::pair<double,std::string> > time_and_name_history;
      time_and_name_history.push_back (std::make_pair (cycle,
                                                       filename_pvtu));
      const std::string filename_pvd (Filename::get_filename_pvd());
      std::ofstream pvd_output (filename_pvd.c_str());
      DataOutBase::write_pvd_record (pvd_output, time_and_name_history);
    }

  if (this_mpi_process == 0)
    convergence_table.write_text(pcout.get_stream());
}

// @sect4{MSP_Toroidal_Membrane::make_grid}

template <int dim>
void MSP_Toroidal_Membrane<dim>::make_grid_manifold_ids ()
{
  // Set refinement manifold to keep geometry of particle
  // as exact as possible
  typename Triangulation<dim>::active_cell_iterator
  cell = triangulation.begin_active(),
  endc = triangulation.end();
  for (; cell!=endc; ++cell)
    {
      const bool cell_1_is_membrane = geometry.within_membrane(cell->center());
      if (!cell_1_is_membrane) continue;

      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
//      if (cell->face(face)->at_boundary()) continue;
//
//      const bool cell_2_is_membrane = geometry.within_membrane(cell->neighbor(face)->center());
//      if (cell_2_is_membrane != cell_1_is_membrane)
          cell->face(face)->set_manifold_id(manifold_id_outer_radius);

//      for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_face; ++vertex)
//      {
//        if (geometry.on_radius_outer(cell->face(face)->vertex(vertex)))
//          cell->face(face)->set_manifold_id(manifold_id_outer_radius);
//
//        if (geometry.on_radius_inner(cell->face(face)->vertex(vertex)))
//          cell->face(face)->set_manifold_id(manifold_id_inner_radius);
//      }
        }
    }
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::make_grid ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Make grid");

  GridIn<dim> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream input (parameters.mesh_file);
  gridin.read_abaqus(input);

  // Set boundary IDs
/*  typename Triangulation<dim>::active_cell_iterator
  cell = triangulation.begin_active(),
  endc = triangulation.end();
  for (; cell!=endc; ++cell)
    {
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          if (cell->face(face)->at_boundary())
            {
              const Point<dim> &centre = cell->face(face)->center();
              if (centre[0] > 0.0 && // Not on the y-axis...
                  centre[0] < geometry.get_membrane_minor_radius_centre()[0] && // ... but to the left of the toroid...
                  centre[1] < geometry.get_torus_minor_radius_outer() && // ...and contained within the height of the toroid
                  centre[1] > -geometry.get_torus_minor_radius_outer())
                {
                  cell->face(face)->set_boundary_id(boundary_id_magnet);
                }
            }
        }
    }
*/

  // Refine adaptively the permanent magnet region for given
  // input parameters of box lenghts
  for(unsigned int cycle = 0; cycle < 1; ++cycle)
  {
      typename Triangulation<dim>::active_cell_iterator
              cell = triangulation.begin_active(),
              endc = triangulation.end();
      for (; cell!=endc; ++cell)
          if(cell->is_locally_owned())
          {
              for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
              {
                  if (std::abs(cell->vertex(vertex)[0]) < parameters.bounding_box_r &&
                      std::abs(cell->vertex(vertex)[1]) < parameters.bounding_box_z)
                  {
                      cell->set_refine_flag();
                      cell->set_material_id(material_id_bar_magnet);
                  }
                  continue;
              }
          }
      triangulation.execute_coarsening_and_refinement();
  }

  // Rescale the geometry before attaching manifolds
  GridTools::scale(parameters.grid_scale, triangulation);

  make_grid_manifold_ids();
  triangulation.set_manifold (manifold_id_outer_radius, manifold_outer_radius);
  triangulation.set_manifold (manifold_id_inner_radius, manifold_inner_radius);

  triangulation.refine_global (parameters.n_global_refinements);

}

// @sect4{MSP_Toroidal_Membrane::run}

template <int dim>
void MSP_Toroidal_Membrane<dim>::run ()
{
  computing_timer.reset();

  for (unsigned int cycle=0; cycle < parameters.n_cycles_max; ++cycle)
    {
      pcout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          make_grid ();
          set_initial_fe_indices ();
        }
      else
        refine_grid ();


      pcout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << " (on "
            << triangulation.n_levels()
            << " levels)"
            << std::endl;

      setup_system ();

      pcout << "   Number of degrees of freedom: "
            << hp_dof_handler.n_dofs()
            << std::endl;

      assemble_system ();
      solve ();
      compute_error ();
      output_results (cycle);
    }
}


// @sect3{The <code>main</code> function}

int main (int argc, char *argv[])
{
  try
    {
      deallog.depth_console (0);
      Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
                                                           numbers::invalid_unsigned_int);
      ConditionalOStream pcout (std::cout,
                                (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

      const std::string input_file ("parameters.prm");

      {
        pcout << "Running with " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " MPI processes" << std::endl;
        const std::string title = "Running in 2-d...";
        const std::string divider (title.size(), '=');

        pcout
            << divider << std::endl
            << title << std::endl
            << divider << std::endl;

        MSP_Toroidal_Membrane<2> msp_toroidal_membrane (input_file);
        msp_toroidal_membrane.run ();
      }

      pcout << std::endl << std::endl;

//      {
//        const std::string title = "Running in 3-d...";
//        const std::string divider (title.size(), '=');

//        pcout
//          << divider << std::endl
//          << title << std::endl
//          << divider << std::endl;

//        MSP_Toroidal_Membrane<3> msp_toroidal_membrane (input_file);
//        msp_toroidal_membrane.run ();
//      }
    }
  catch (std::exception &exc)
    {
//    for (unsigned int i=0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
//    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl
//            << "--- PROCESS " << i << "---"
                    << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "Exception on processing: " << std::endl
                    << exc.what() << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
        }
//      MPI_Barrier(MPI_COMM_WORLD);
//    }

      return 1;
    }
  catch (...)
    {
//    for (unsigned int i=0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
//    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cerr << std::endl << std::endl
                    << "----------------------------------------------------"
                    << std::endl
//            << "--- PROCESS " << i << "---"
                    << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
          std::cerr << "Unknown exception!" << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------"
                    << std::endl;
        }
//      MPI_Barrier(MPI_COMM_WORLD);
//    }
      return 1;
    }
//  catch (std::exception &exc)
//    {
//      std::cerr << std::endl << std::endl
//                << "----------------------------------------------------"
//                << std::endl;
//      std::cerr << "Exception on processing: " << std::endl
//                << exc.what() << std::endl
//                << "Aborting!" << std::endl
//                << "----------------------------------------------------"
//                << std::endl;
//
//      return 1;
//    }
//  catch (...)
//    {
//      std::cerr << std::endl << std::endl
//                << "----------------------------------------------------"
//                << std::endl;
//      std::cerr << "Unknown exception!" << std::endl
//                << "Aborting!" << std::endl
//                << "----------------------------------------------------"
//                << std::endl;
//      return 1;
//    }

  return 0;
}
