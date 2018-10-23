#include <MSP_toroidal_membrane.h>

using namespace dealii;

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
  manifold_id_magnet(3),
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
                                 parameters.mu_r_membrane),
  phi_fe(phi_component),
  u_fe(u_componenent),
  dofs_per_block(n_blocks)
{
  AssertThrow(parameters.poly_degree_max >= parameters.poly_degree_min, ExcInternalError());

  for (unsigned int degree = parameters.poly_degree_min;
       degree <= parameters.poly_degree_max; ++degree)
    {
      degree_collection.push_back(degree); // Polynomial degree
//      fe_collection.push_back(FE_Q<dim>(degree));
      fe_collection.push_back(FESystem<dim>(FE_Q<dim>(degree), 1, // scalar fe for magnetic potential
                                            FE_Q<dim>(degree), dim)); // vector fe for displacement
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

//      if (geometry.within_membrane(cell->center()))
//        cell->set_active_fe_index(0); // 1 for p-refinement test

    // Setting of higher degree FE to cells in toroid membrane
    if (cell->material_id() == 1)
        cell->set_active_fe_index(0); // 1 for FE_Q(2) or 2 for FE_Q(3)
    else
        cell->set_active_fe_index(0);
    }
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::make_constraints (ConstraintMatrix &constraints, const int &itr_nr)
{
    // All dirichlet constraints need to be specified only at 0th NR iteration
    // constraints are different at different NR iterations
    constraints.clear();
    const bool apply_dirichlet_bc = (itr_nr == 0); // need to apply inhomogeneous DBC

    // Scalar extractor for components of vector displacement field
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);

    // applying inhomogeneous DBC at the 0th NR iteration
    if(apply_dirichlet_bc)
    {
        // applying inhomogeneous DBC for the scalar magnetic potential field
        std::map< types::global_dof_index, Point<dim> > support_points;
        DoFTools::map_dofs_to_support_points(mapping_collection, hp_dof_handler, support_points);
        LinearScalarPotential<dim> linear_scalar_potential(parameters.potential_difference_per_unit_length);

        for(auto it : support_points)
        {
            const auto dof_index = it.first;
            const auto supp_point = it.second;

            // Check for the support point if inside the permanent magnet region:
            // In 2D axisymmetric we have x,y <=> r,z so need to compare 0th and 1st component of point
            // In 3D we have x,y,z <=> r,z,theta so need to compare 0th and 1st component of point
            if( std::abs(supp_point[0]) <= parameters.bounding_box_r && // X coord of support point less than magnet radius...
                std::abs(supp_point[1]) <= parameters.bounding_box_z && // Y coord of support point less than magnet height
                (dim == 3 ? std::abs(supp_point[2]) <= parameters.bounding_box_r : true) && // Z coord
                (dim == 3 ?
                 std::hypot(supp_point[0], supp_point[2]) < (0.98 * parameters.bounding_box_r) // radial distance on XZ plane with tol of 2%
                 : true))
            {
    //            pcout << "DoF index: " << dof_index << "    " << "point: " << supp_point << std::endl;
                const double potential_value = linear_scalar_potential.value(supp_point);
    //            pcout << "Potential value: " << potential_value << std::endl;
                constraints.add_line(dof_index);
                constraints.set_inhomogeneity(dof_index, potential_value);
            }
        }

        // applying the inhomogeneous DBC for the vector valued displacement field
        {
            // zero DBC on left boundary (0th face) i.e. u_x = u_y = u_z = 0
            const int boundary_id = 0;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe_collection.component_mask(u_fe));
        }
        {
            // zero DBC on right boundary (face = 1) i.e. u_x = 0
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe_collection.component_mask(x_displacement));
        }
        {
            // inhomogeneous DBC on right boundary (face = 1) i.e. u_y != 0
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ConstantFunction<dim>(-0.1, n_components), // considering a strain of 10%
                                                     constraints,
                                                     fe_collection.component_mask(y_displacement));
        }
        if(dim == 3)
        {
            // zero DBC on right boundary (face = 1) i.e. u_z = 0
            const FEValuesExtractors::Scalar z_displacement(2);
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe_collection.component_mask(z_displacement));
        }
    }
    else // apply homogeneous DBC to the previously inhomogenoeus DBC constrained DoFs
    {
        // set homogeneous DBC for scalar magnetic potential field at itr_nr > 0
        std::map< types::global_dof_index, Point<dim> > support_points;
        DoFTools::map_dofs_to_support_points(mapping_collection, hp_dof_handler, support_points);
        Functions::ZeroFunction<dim> zero_function(1);

        for(auto it : support_points)
        {
            const auto dof_index = it.first;
            const auto supp_point = it.second;

            // Check for the support point if inside the permanent magnet region:
            // In 2D axisymmetric we have x,y <=> r,z so need to compare 0th and 1st component of point
            // In 3D we have x,y,z <=> r,z,theta so need to compare 0th and 1st component of point
            if( std::abs(supp_point[0]) <= parameters.bounding_box_r && // X coord of support point less than magnet radius...
                std::abs(supp_point[1]) <= parameters.bounding_box_z && // Y coord of support point less than magnet height
                (dim == 3 ? std::abs(supp_point[2]) <= parameters.bounding_box_r : true) && // Z coord
                (dim == 3 ?
                 std::hypot(supp_point[0], supp_point[2]) < (0.98 * parameters.bounding_box_r) // radial distance on XZ plane with tol of 2%
                 : true))
            {
                const double potential_value = zero_function.value(supp_point);
                constraints.add_line(dof_index);
                constraints.set_inhomogeneity(dof_index, potential_value);
            }
        }

        // set homogeneous DBC for the vector valued displacement field at itr_nr > 0
        {
            // zero DBC on left boundary (0th face) i.e. u_x = u_y = u_z = 0
            const int boundary_id = 0;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components), // all dim components of displ
                                                     constraints,
                                                     fe_collection.component_mask(u_fe));
        }
        {
            // zero DBC on right boundary (face = 1) i.e. u_x = 0
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe_collection.component_mask(x_displacement));
        }
        {
            // set homogeneous DBC on right boundary (face = 1) i.e. u_y = 0 for itr_nr > 0
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe_collection.component_mask(y_displacement));
        }
        if(dim == 3)
        {
            // zero DBC on right boundary (face = 1) i.e. u_z = 0
            const FEValuesExtractors::Scalar z_displacement(2);
            const int boundary_id = 1;
            VectorTools::interpolate_boundary_values(hp_dof_handler,
                                                     boundary_id,
                                                     Functions::ZeroFunction<dim>(n_components),
                                                     constraints,
                                                     fe_collection.component_mask(z_displacement));
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

    std::vector<unsigned int> block_component (n_components, u_block); // displacement
    block_component[phi_component] = phi_block; // magnetic scalar potential
    // Distribute DoFs
    hp_dof_handler.distribute_dofs (fe_collection);
   // When using parallel::shared::Triangulation no need to do this
//    DoFRenumbering::subdomain_wise (hp_dof_handler);
    DoFRenumbering::Cuthill_McKee (hp_dof_handler);
    DoFRenumbering::component_wise(hp_dof_handler, block_component);
    DoFTools::count_dofs_per_block(hp_dof_handler, dofs_per_block, block_component);

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
    hanging_node_constraints.clear();
    // make_hanging_node_constraints needs locally owned and ghost cells. Cannot do it with
    // normal triangulation with subdomain is indicator of locally owned cells
//    constraints.reinit(locally_relevant_dofs);

    // for now just setup the hanging node constraints for current ref cycle
    DoFTools::make_hanging_node_constraints (hp_dof_handler,
                                             hanging_node_constraints);

    // Will have to setup the dirichlet constraints for vector valued solution field
    // in the nonlinear solver part, i.e. for each adaptive ref cycle for each NR iter of
    // the current load or time increment cycle
//    make_constraints(constraints);
    hanging_node_constraints.close();
    constraints.close ();
    // merge both constraints matrices with dbc constraints dominating when conflict occurs on same dof
//    constraints.merge(hanging_node_constraints,  ConstraintMatrix::MergeConflictBehavior::left_object_wins);
  }

  {
    TimerOutput::Scope timer_scope (computing_timer, "Setup: matrix, vectors");

    std::vector<dealii::types::global_dof_index> n_locally_owned_dofs_per_processor (n_mpi_processes);
    {
      AssertThrow(all_locally_owned_dofs.size() == n_locally_owned_dofs_per_processor.size(), ExcInternalError());
      for (unsigned int i=0; i < n_locally_owned_dofs_per_processor.size(); ++i)
        n_locally_owned_dofs_per_processor[i] = all_locally_owned_dofs[i].n_elements();
    }

    const unsigned int n_phi = dofs_per_block[0];
    const unsigned int n_u = dofs_per_block[1];
    locally_owned_partitioning.clear();
    locally_relevant_partitioning.clear();
    locally_owned_partitioning.push_back(locally_owned_dofs.get_view(0,n_phi));
    locally_owned_partitioning.push_back(locally_owned_dofs.get_view(n_phi, n_phi + n_u));
    locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(0,n_phi));
    locally_relevant_partitioning.push_back(locally_relevant_dofs.get_view(n_phi, n_phi + n_u));

    TrilinosWrappers::BlockSparsityPattern sp (locally_owned_partitioning,
                                               locally_owned_partitioning,
                                               locally_relevant_partitioning,
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
    /*system_rhs.reinit(locally_owned_dofs,
                      mpi_communicator);
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);*/
    system_rhs.reinit(locally_owned_partitioning, mpi_communicator);
    solution.reinit(locally_owned_partitioning, locally_relevant_partitioning, mpi_communicator);
    setup_quadrature_point_history();
  }
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::setup_quadrature_point_history()
{
    pcout << "Setting up quadrature point data..." << std::endl;

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
      if (cell->is_locally_owned())
      {
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
        const unsigned int  &n_q_points = fe_values.n_quadrature_points;
        quadrature_point_history.initialize(cell, n_q_points);
      }

    for (cell = hp_dof_handler.begin_active(); cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
          hp_fe_values.reinit(cell);
          const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
          const unsigned int  &n_q_points = fe_values.n_quadrature_points;
          const std::vector<std::shared_ptr<PointHistory<dim,dim_Tensor> > > lqph =
                  quadrature_point_history.get_data(cell);
          Assert(lqph.size() == n_q_points, ExcInternalError());

          for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
              Assert(lqph[q_point], ExcInternalError());
              lqph[q_point]->setup_lqp(parameters);
          }
      }
}

template <int dim>
void
MSP_Toroidal_Membrane<dim>::update_qph_incremental(const TrilinosWrappers::MPI::BlockVector &solution_delta)
{
    TimerOutput::Scope timer_scope (computing_timer, "Update QPH data");
    pcout << "Update QPH data" << std::endl;

    TrilinosWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                      locally_relevant_partitioning,
                                                      mpi_communicator);
    solution_total = get_total_solution(solution_delta);
    std::vector<Tensor<2, dim> > solution_grads_u_total;
    std::vector<Tensor<1, dim> > solution_values_u_total;
    std::vector<double> solution_values_phi_total;

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
        if(cell->is_locally_owned())
        {
            hp_fe_values.reinit(cell);
            const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
            const unsigned int  &n_q_points = fe_values.n_quadrature_points;
            const std::vector<Point<dim> > &quadrature_points = fe_values.get_quadrature_points();

            solution_grads_u_total.clear();
            solution_values_u_total.clear();
            solution_values_phi_total.clear();
            solution_grads_u_total.resize(n_q_points, Tensor<2, dim>());
            solution_values_u_total.resize(n_q_points, Tensor<1, dim>());
            solution_values_phi_total.resize(n_q_points);

            const std::vector<std::shared_ptr<PointHistory<dim,dim_Tensor> > > lqph =
                    quadrature_point_history.get_data(cell);
            Assert(lqph.size() == n_q_points, ExcInternalError());

            fe_values[u_fe].get_function_gradients(solution_total,
                                                   solution_grads_u_total);
            fe_values[u_fe].get_function_values(solution_total,
                                                solution_values_u_total);
            fe_values[phi_fe].get_function_values(solution_total,
                                                  solution_values_phi_total);

            // need to apply transformation here before sending the soln grads u total
            // to transform 2*2 tensor to 3*3 since it is used to calculate F

            std::vector<Tensor<2, dim_Tensor> > solution_grads_u_total_transformed(n_q_points,
                                                                                   Tensor<2, dim_Tensor>());
            // if axisymmetric simulation, need this transformation of
            // gradient of 2D shape functions to get dim 3 gradient of shape functions Tensor
            // accounting for the circumferential strain i.e. hoop stress
            if(dim == 2)
            {
                for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    // Get the x co-ord to the quadrature point
                    const double radial_distance = quadrature_points[q_point][0];

                    // copy dim 2 tensor components into corresponding dim 3 tensor
                    // leaving dim 3 Tensor components 0,2 = 1,2 = 2,0 = 2,1 = 0
                    /*      dim 2 mapped to   dim 3 Tensor
                     * | u_r,r  u_r,z | ->  | u_r,r  u_r,z   0    |
                     * | u_z,r  u_z,z | ->  | u_z,r  u_z,z   0    |
                     *                      | 0      0      u_r/R |
                     * */
                    for(unsigned int i = 0; i < dim; ++i)
                        for(unsigned int j = 0; j < dim; ++j)
                    {
                        solution_grads_u_total_transformed[q_point][i][j] = solution_grads_u_total[q_point][i][j];
                    }
                    // u_theta,theta = u_r / R
                    solution_grads_u_total_transformed[q_point][dim][dim] = solution_values_u_total[q_point][0] / radial_distance;
                }

                for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    Assert(lqph[q_point], ExcInternalError());
                    lqph[q_point]->update_values(solution_grads_u_total_transformed[q_point],
                                                 solution_values_phi_total[q_point]);
                }
            }
            // for 3D simulation proceed normally
  /*          else if(dim == 3)
            {
                for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                    lqph[q_point]->update_values(solution_grads_u_total[q_point],
                                                 solution_values_phi_total[q_point]);
            }*/
            else
                Assert(false, ExcInternalError());
        }
}

// Return total solution known by this MPI process
template <int dim>
TrilinosWrappers::MPI::BlockVector
MSP_Toroidal_Membrane<dim>::get_total_solution(const TrilinosWrappers::MPI::BlockVector &solution_delta) const
{
    TrilinosWrappers::MPI::BlockVector solution_total(solution);
    solution_total += solution_delta;
    return solution_total;
}

// @sect4{MSP_Toroidal_Membrane::assemble_system}

template <int dim>
void MSP_Toroidal_Membrane<dim>::assemble_system ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Assembly");

  system_matrix = 0.0;
  system_rhs = 0.0;

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

      const std::vector<std::shared_ptr<PointHistory<dim,dim_Tensor> > > lqph =
              quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      // shape function values for displacement component
      std::vector<std::vector<Tensor<1, dim> > > Nx (n_q_points,
                                                     std::vector<Tensor<1, dim> >(n_dofs_per_cell));
      std::vector<std::vector<Tensor<2, dim> > > Grad_Nx(n_q_points,
                                                         std::vector<Tensor<2, dim> >(n_dofs_per_cell));
      std::vector<std::vector<Tensor<2, dim_Tensor> > > Grad_Nx_transformed(n_q_points,
                                                                            std::vector<Tensor<2, dim_Tensor> >(n_dofs_per_cell));
      std::vector<std::vector<SymmetricTensor<2, dim_Tensor> > > dE(n_q_points,
                                                                    std::vector<SymmetricTensor<2, dim_Tensor> >(n_dofs_per_cell));

      for(unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
          Assert(lqph[q_index], ExcInternalError());
          const Tensor<2, dim_Tensor> F_inv = lqph[q_index]->get_F_inv();
          Tensor<2, dim_Tensor> F = invert(F_inv);

          for(unsigned int k = 0; k < n_dofs_per_cell; ++k)
          {
              const unsigned int k_group = fe_values.get_fe().system_to_base_index(k).first.first;

              if(k_group == u_block)
              {
                  Nx[q_index][k] = fe_values[u_fe].value(k, q_index);
                  Grad_Nx[q_index][k] = fe_values[u_fe].gradient(k, q_index);

                  // Need to apply some transformation here
                  // Grad_Nx is 2*2 dim tensor but F is 3*3 dim tensor
                  if(dim == 2)
                  {
                      // copy dim 2 tensor components into corresponding dim 3 tensor
                      // leaving dim 3 Tensor components 0,2 = 1,2 = 2,0 = 2,1 = 0
                      /*      dim 2 mapped to   dim 3 Tensor
                       * | u_r,r  u_r,z | ->  | u_r,r  u_r,z   0    |
                       * | u_z,r  u_z,z | ->  | u_z,r  u_z,z   0    |
                       *                      | 0      0      u_r/R |
                       * */
                      for(unsigned int i = 0; i < dim; ++i)
                          for(unsigned int j = 0; j < dim; ++j)
                              Grad_Nx_transformed[q_index][k][i][j] = Grad_Nx[q_index][k][i][j];

                      const double radial_distance = quadrature_points[q_index][0];
                      Grad_Nx_transformed[q_index][k][dim][dim] = Nx[q_index][k][0] /  radial_distance;

                      Tensor<2, dim_Tensor> temp = transpose(F) * Grad_Nx_transformed[q_index][k];
                      dE[q_index][k] = symmetrize(temp); // variation or increment of Green-Lagrange strain
                  }
                  // for 3D simulation proceed normally
                /*  else if(dim == 3)
                  {
                      Tensor<2, dim_Tensor> temp = transpose(F) * Grad_Nx[q_index][k];
                      dE[q_index][k] = symmetrize(temp); // variation or increment of Green-Lagrange strain
                  }*/
                  else
                      Assert(false, ExcInternalError());
              }
              else
                  Assert(k_group <= u_block, ExcInternalError());
          }
      }

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          Assert(lqph[q_index], ExcInternalError());
          const Tensor<2, dim_Tensor> S = lqph[q_index]->get_second_Piola_Kirchoff_stress();
          const SymmetricTensor<4, dim_Tensor> C = lqph[q_index]->get_4th_order_material_elasticity();

          const double mu_r_mu_0 = coefficient_values[q_index];
          // Get the x co-ord to the quadrature point
          const double radial_distance = quadrature_points[q_index][0];
          // If dim == 2, assembly using axisymmetric formulation
          const double coord_transformation_scaling = ( dim == 2
                                                        ?
                                                          2.0 * dealii::numbers::PI * radial_distance
                                                        :
                                                          1.0);

          // Assemble system matrix aka tangent matrix
          for (unsigned int i=0; i<n_dofs_per_cell; ++i)
            {
              const unsigned int component_i = fe_values.get_fe().system_to_component_index(i).first;
              const unsigned int i_group = fe_values.get_fe().system_to_base_index(i).first.first;

              for (unsigned int j=0; j<=i; ++j)
              {
                  const unsigned int component_j = fe_values.get_fe().system_to_component_index(j).first;
                  const unsigned int j_group = fe_values.get_fe().system_to_base_index(j).first.first;

                  // K_uu contribution: comprising of material and geometrical stress contribution
                  if((i_group == j_group) && (i_group == u_block))
                  {
                      // material contribution
                      // dE: variation or increment of Green-Lagrange strain
                      // C: 4th order material elasticity tensor
                      cell_matrix(i,j) += dE[q_index][i] * C
                                          * dE[q_index][j] * fe_values.JxW(q_index);

                      // Add geometrical stress contribution to local matrix diagonals only
                      if(component_i == component_j)
                      {
                          // DdE: Linearisation of increment of Green-Lagrange strain tensor
                          // S: second Piola-Kirchoff stress tensor

                          // Need to apply some transformation here
                          // to get the resulting DdE tensor of 3*3 dim from
                          // grad of shape functions which are 2*2 tensor for axisymmetric formulation
                          if(dim == 2)
                          {
                              const SymmetricTensor<2, dim_Tensor, double> DdE_ij = symmetrize(
                                                                         transpose(Grad_Nx_transformed[q_index][i]) *
                                                                         Grad_Nx_transformed[q_index][j]);

                              cell_matrix(i,j) += scalar_product(DdE_ij, S) * fe_values.JxW(q_index);
                          }
                          // for 3D simulation proceed normally
                         /* else if(dim == 3)
                          {
                              const SymmetricTensor<2, dim_Tensor, double> DdE_ij = symmetrize(
                                                                         transpose(Grad_Nx[q_index][i]) *
                                                                         Grad_Nx[q_index][j]);

                              cell_matrix(i,j) += scalar_product(DdE_ij, S) * fe_values.JxW(q_index);
                          }*/
                          else
                              Assert(false, ExcInternalError());
                      }
                  }

                  // Purely magnetic contributions K_phiphi
                  else if((i_group == j_group) && (i_group == phi_block))
                  {
                      cell_matrix(i,j) += fe_values[phi_fe].gradient(i,q_index) *
                                          mu_r_mu_0*
                                          coord_transformation_scaling *
                                          fe_values[phi_fe].gradient(j,q_index) *
                                          fe_values.JxW(q_index);
                  }
                  else
                      Assert((i_group <= u_block) && (j_group <= u_block),
                             ExcInternalError());
              }
            }

          // Assemble RHS vector
          for (unsigned int i=0; i<n_dofs_per_cell; ++i)
          {
              const unsigned int i_group = fe_values.get_fe().system_to_base_index(i).first.first;

              // RHS is negative residual term
              if (i_group == u_block)
                  cell_rhs(i) -= scalar_product(S, dE[q_index][i]) * fe_values.JxW(q_index);
              else if (i_group == phi_block)
                  cell_rhs(i) -= 0.0;
              else
                  Assert(i_group <= u_block, ExcInternalError());
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
                                              system_rhs,
                                              true);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


// @sect4{MSP_Toroidal_Membrane::solve}

template <int dim>
void MSP_Toroidal_Membrane<dim>::solve (TrilinosWrappers::MPI::BlockVector &newton_update)
{
  TimerOutput::Scope timer_scope (computing_timer, "Solve linear system");

  TrilinosWrappers::MPI::BlockVector distributed_solution(locally_owned_partitioning,
                                                          mpi_communicator);
  distributed_solution = newton_update;

  // Block to solve for: either displacement block or magnetic scalar potential block
  // will have to change for a coupled problem in future
  unsigned int solution_block;
  if(parameters.problem_type == "Purely magnetic")
      solution_block = phi_block;
  else if (parameters.problem_type == "Purely elastic")
      solution_block = u_block;
  else
      Assert(false, ExcMessage("Coupled linear solver not implemented!"));

  // Need to update for considered block we are solving? Eg. system_matrix.block(u_block,  u_block).m()
  // and similar way for linear solver tolerance?
  SolverControl solver_control (parameters.lin_slvr_max_it*system_matrix.block(solution_block, solution_block).m(),
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

          ptr_prec->initialize(system_matrix.block(solution_block,solution_block),
                               additional_data);
          preconditioner.reset(ptr_prec);
        }
      else if (parameters.preconditioner_type == "ssor")
        {
          TrilinosWrappers::PreconditionSSOR *ptr_prec
            = new TrilinosWrappers::PreconditionSSOR ();

          TrilinosWrappers::PreconditionSSOR::AdditionalData
          additional_data (parameters.preconditioner_relaxation);

          ptr_prec->initialize(system_matrix.block(solution_block,solution_block),
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
          ptr_prec->initialize(system_matrix.block(solution_block,solution_block),
                               additional_data);
          preconditioner.reset(ptr_prec);
        }

      solver.solve (system_matrix.block(solution_block,solution_block),
                    distributed_solution.block(solution_block),
                    system_rhs.block(solution_block),
                    *preconditioner);
    }
  else // Direct
    {
      TrilinosWrappers::SolverDirect solver (solver_control);
      solver.solve (system_matrix.block(solution_block,solution_block),
                    distributed_solution.block(solution_block),
                    system_rhs.block(solution_block));
    }

  constraints.distribute (distributed_solution);
  newton_update = distributed_solution;

  pcout
      << "  Linear solver iterations: " << solver_control.last_step()
      << "\tLinear solver residual: " << solver_control.last_value()
      << std::endl;
}

// Newton-Raphson scheme to solve nonlinear system of equation iteratively
template <int dim>
void
MSP_Toroidal_Membrane<dim>::solve_nonlinear_system(TrilinosWrappers::MPI::BlockVector &solution_delta)
{
    TrilinosWrappers::MPI::BlockVector newton_update(locally_owned_partitioning,
                                                     locally_relevant_partitioning,
                                                     mpi_communicator);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_convergence_header();

    unsigned int newton_iteration = 0;
    for(; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
    {
        pcout << "  Newton iteration:  " << newton_iteration << "  " <<std::endl;

        // since we use NR scheme to solve the fully nonlinear problem
        // data stored in tangent matrix and RHS vector is not reusable so clear
        system_matrix = 0.0;
        system_rhs = 0.0;

        // impose the DBC for displacement
        make_constraints(constraints, newton_iteration); // need to update the function for
        // constraining displacement dofs separately and the scalar magnetic potential separately
        constraints.close();

        // merge both constraints matrices with hanging node constraints dominating when conflict occurs on same dof
        constraints.merge(hanging_node_constraints,  ConstraintMatrix::MergeConflictBehavior::right_object_wins);
//        constraints.condense(system_matrix, system_rhs); // need to check this for MPI::BlockVector

        assemble_system();
        get_error_residual(error_residual);

        if (newton_iteration == 0)
            error_residual_0 = error_residual;

        // find the normalized error in residual and check for convergence
        error_residual_norm = error_residual;
        error_residual_norm.normalize(error_residual_0);

        if (newton_iteration > 0 && error_residual_norm.u <= parameters.tol_f
                && error_update_norm.u <= parameters.tol_u)
        {
            pcout << " CONVERGED!" << std::endl;
            print_convergence_footer();
            break;
        }

        // Solve linear system
        solve(newton_update);

        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
            error_update_0 = error_update;

        // Find the normalized error in newton update
        error_update_norm = error_update;
        error_update_norm.normalize(error_update_0);

        // update the solution with current solution increment
        solution_delta += newton_update;
        // update qph related to this new displacement and stress state
        update_qph_incremental(solution_delta);

        pcout << std::fixed << std::setprecision(3) << std::scientific
              << error_residual_norm.norm << "  "
              << error_residual_norm.u << "  "
              << error_residual_norm.phi << "  "
              << error_update_norm.norm << "  "
              << error_update_norm.u << "  "
              << error_update_norm.phi << "\n" << std::endl;
    }
    // if more NR iterations performed than max allowed
    AssertThrow (newton_iteration < parameters.max_iterations_NR,
                 ExcMessage("No convergence in nonlinear solver!"));
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::print_convergence_header()
{
    static const unsigned int l_width = 155;

    for(unsigned int i = 0; i < l_width; ++i)
        pcout << "-";
    pcout << std::endl;

    pcout << "RES_NORM  "
          << "RES_U  RES_PHI  NU_NORM  "
          << "NU_U  NU_PHI  " << std::endl;

    for(unsigned int i = 0; i < l_width; ++i)
        pcout << "-";
    pcout << std::endl;
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::print_convergence_footer()
{
    static const unsigned int l_width = 155;

    for(unsigned int i = 0; i < l_width; ++i)
        pcout << "-";
    pcout << std::endl;

    pcout << "Relative errors:" << std::endl
          << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
          << "Force:\t\t" << error_residual.u / error_residual_0.u << std::endl;
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

  TrilinosWrappers::MPI::BlockVector distributed_solution(locally_owned_partitioning,
                                                          mpi_communicator);
  distributed_solution = solution;
  const BlockVector<double> localised_solution(distributed_solution);

  // --- Kelly Error estimator ---
  estimated_error_per_cell.reinit(triangulation.n_active_cells());
  estimated_error_per_cell = 0.0;
  KellyErrorEstimator<dim>::estimate (hp_dof_handler,
                                      EE_qf_collection_face_QGauss,
                                      typename FunctionMap<dim>::type(),
                                      localised_solution,
                                      estimated_error_per_cell,
                                      fe_collection.component_mask(u_fe));
//                                      ComponentMask());
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

// Class to output gradients of magnetic scalar potential from the solution vector
// i.e. magnetic field h
// Input: solution block corresponding to magnetic scalar potential (scalar field)
// Output: gradient of magnetic scalar potential (vector field)
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

// Class to output displacements from the solution vector
// Input: solution block corresponding to displacement field (vector field)
// Output: displacement (vector field)
template<int dim>
class DisplacementFieldPostprocessor : public DataPostprocessorVector<dim>
{
public:
    DisplacementFieldPostprocessor()
        : DataPostprocessorVector<dim> ("displacement", update_values)
    {}

    virtual ~DisplacementFieldPostprocessor() {}

    virtual void
    evaluate_vector_field (const DataPostprocessorInputs::Vector<dim> &input_data,
                           std::vector<Vector<double> >               &computed_quantities) const
    {
        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert(input_data.solution_gradients.size() == n_quadrature_points,
               ExcInternalError());
        Assert(computed_quantities.size() == n_quadrature_points,
               ExcInternalError());
        for(unsigned int q = 0; q < n_quadrature_points; ++q)
        {
            for(unsigned int d = 0; d < dim; ++d)
            {
                computed_quantities[q](d) = input_data.solution_values[q](d);
            }
        }
    }
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

  std::vector<std::string> solution_names (1, "magnetic_scalar_potential");
  solution_names.emplace_back("displacement");
  if (dim >= 2)
      solution_names.emplace_back("displacement");
  if (dim == 3)
      solution_names.emplace_back("displacement");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  data_component_interpretation(1, DataComponentInterpretation::component_is_scalar);
  data_component_interpretation.emplace_back(DataComponentInterpretation::component_is_part_of_vector);
  if (dim >= 2)
      data_component_interpretation.emplace_back(DataComponentInterpretation::component_is_part_of_vector);
  if (dim == 3)
      data_component_interpretation.emplace_back(DataComponentInterpretation::component_is_part_of_vector);

//  MagneticFieldPostprocessor<dim> mag_field_bar_magnet(material_id_bar_magnet);
//  MagneticFieldPostprocessor<dim> mag_field_toroid(material_id_toroid); // Material ID for Toroid tube as read in from Mesh file
//  MagneticFieldPostprocessor<dim> mag_field_vacuum(material_id_vacuum); // Material ID for free space
  DisplacementFieldPostprocessor<dim> displacements;
  FilteredDataOut< dim,hp::DoFHandler<dim> > data_out (this_mpi_process);

  data_out.attach_dof_handler (hp_dof_handler);

  data_out.add_data_vector (solution, displacements);
 /* data_out.add_data_vector (hp_dof_handler,
                            solution, solution_names,
                            data_component_interpretation);*/
//  data_out.add_data_vector (estimated_error_per_cell, "estimated_error");
  /*data_out.add_data_vector (solution, mag_field_bar_magnet);
  data_out.add_data_vector (solution, mag_field_toroid);
  data_out.add_data_vector (solution, mag_field_vacuum);*/

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
//      const bool cell_1_is_membrane = geometry.within_membrane(cell->center());
//      if (!cell_1_is_membrane) continue;

//      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//      {
    //      if (cell->face(face)->at_boundary()) continue;
    //
    //      const bool cell_2_is_membrane = geometry.within_membrane(cell->neighbor(face)->center());
    //      if (cell_2_is_membrane != cell_1_is_membrane)
//              cell->face(face)->set_manifold_id(manifold_id_outer_radius);

    //      for (unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_face; ++vertex)
    //      {
    //        if (geometry.on_radius_outer(cell->face(face)->vertex(vertex)))
    //          cell->face(face)->set_manifold_id(manifold_id_outer_radius);
    //
    //        if (geometry.on_radius_inner(cell->face(face)->vertex(vertex)))
    //          cell->face(face)->set_manifold_id(manifold_id_inner_radius);
    //      }
//      }

      if(cell->material_id() == 1)
      {
          for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
          {
              if(cell->neighbor(face)->material_id() != cell->material_id())
              {
                  cell->face(face)->set_manifold_id(manifold_id_outer_radius);
              }
          }
      }
    }
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::make_grid ()
{
  TimerOutput::Scope timer_scope (computing_timer, "Make grid");

  // Make rectangular beam for finite strain elasticity problem test
  if(parameters.geometry_shape == "Beam")
  {
      std::vector<unsigned int> repetitions;
      repetitions.push_back(8); // 8 blocks in x direction
      if(dim >= 2)
          repetitions.push_back(4); // 4 blocks in y direction
      if(dim >= 3)
          repetitions.push_back(1); // 1 block in z direction

      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                (dim == 3 ? Point<dim>(0.0, 0.0, 0.0) : Point<dim>(0.0, 0.0)),
                                                (dim == 3 ? Point<dim>(2.0, 1.0, 0.25) : Point<dim>(2.0, 1.0)),
                                                true); // set colorize for boundary ids

      // Rescale the geometry before attaching manifolds
      GridTools::scale(parameters.grid_scale, triangulation);

      triangulation.refine_global (parameters.n_global_refinements);
  }

  // For our geometry of interest for coupled problem
  // use the torus membrane geometry read in from the mesh file
  else if(parameters.geometry_shape == "Toroidal_tube")
  {
      GridIn<dim> gridin;
      gridin.attach_triangulation(triangulation);
      std::ifstream input (parameters.mesh_file); // Use for production code
    //  std::ifstream input (std::string(SOURCE_DIR + parameters.mesh_file)); // Use for testing the code with ctest
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

      // Attach manifold to the cells within the permanent magnet region
      if(dim == 3)
      {
          CylindricalManifold<dim>   manifold_cylindrical(1);
          for (const auto &cell : triangulation.active_cell_iterators())
          {
              // For a block of cells at center within a torous region of rectangular cross section
              const auto cell_center = cell->center();
              if(std::hypot(cell_center[0], cell_center[2]) >= 0.17 && // inner (min) radial distance
                 std::hypot(cell_center[0], cell_center[2]) <= 0.27 && // outer (max) radial distance
                 std::abs(cell_center[1]) <= 0.06) // max axial height
              {
                  cell->set_all_manifold_ids(manifold_id_magnet);
              }
              if(std::hypot(cell_center[0], cell_center[2]) < 0.17 &&
                 std::abs(cell_center[1]) <= 0.06)
              {
                  cell->set_all_manifold_ids(5);
              }
          }
          triangulation.set_manifold(manifold_id_magnet, manifold_cylindrical);

          manifold_magnet.initialize(triangulation);
          triangulation.set_manifold(5, manifold_magnet);
      }

      // Refine adaptively the permanent magnet region for given
      // input parameters of box lenghts
      for(unsigned int cycle = 0; cycle < 2; ++cycle)
      {
          typename Triangulation<dim>::active_cell_iterator
                  cell = triangulation.begin_active(),
                  endc = triangulation.end();
          for (; cell!=endc; ++cell)
          {
              for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
              {
                  if (std::abs(cell->vertex(vertex)[0]) < parameters.bounding_box_r &&
                      std::abs(cell->vertex(vertex)[1]) < parameters.bounding_box_z &&
                      (dim == 3 ? std::abs(cell->vertex(vertex)[2]) < parameters.bounding_box_r : true))
                  {
                      cell->set_refine_flag();
                  }
                  continue;
              }
          }
          triangulation.execute_coarsening_and_refinement();
      }

      // Set material id to bar magnet for the constrained cells
      typename Triangulation<dim>::active_cell_iterator
              cell = triangulation.begin_active(),
              endc = triangulation.end();
      for (; cell!=endc; ++cell)
      {
          unsigned int vertex_count = 0;
          for (unsigned int vertex = 0; vertex < GeometryInfo<dim>::vertices_per_cell; ++vertex)
          {
              if (std::abs(cell->vertex(vertex)[0]) <= parameters.bounding_box_r &&
                  std::abs(cell->vertex(vertex)[1]) <= parameters.bounding_box_z &&
                  (dim == 3 ? std::abs(cell->vertex(vertex)[2]) <= parameters.bounding_box_r : true) &&
                  (dim == 3 ?
                   std::hypot(cell->vertex(vertex)[0],cell->vertex(vertex)[2]) < (0.98 * parameters.bounding_box_r) //radial distance with tolerance of 2%
                   :
                   true))
                  vertex_count++;
          }

          if(vertex_count >= (GeometryInfo<dim>::vertices_per_cell))
              cell->set_material_id(material_id_bar_magnet);
      }

      // Rescale the geometry before attaching manifolds
      GridTools::scale(parameters.grid_scale, triangulation);

      make_grid_manifold_ids();

      if(dim == 2) // Use Spherical Manifold for torous membrane
      {
          triangulation.set_manifold (manifold_id_outer_radius, manifold_outer_radius);
          triangulation.set_manifold (manifold_id_inner_radius, manifold_inner_radius);
      }
      if(dim == 3) // Use Torus Manifold for torus membrane
      {
          TorusManifold<dim> manifold_torus_outer_radius (geometry.get_membrane_minor_radius_centre()[0],
                                                          geometry.get_torus_minor_radius_outer());
    //      triangulation.set_manifold(manifold_id_outer_radius, manifold_torus_outer_radius);
      }

      triangulation.refine_global (parameters.n_global_refinements);

  }

  else
      AssertThrow(false, ExcInternalError());
}

template <int dim>
void MSP_Toroidal_Membrane<dim>::postprocess_energy()
{
    TimerOutput::Scope timer_scope (computing_timer, "Postprocess energy");

    hp::FEValues<dim> hp_fe_values (mapping_collection,
                                    fe_collection,
                                    qf_collection_cell,
                                    update_values |
                                    update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);

    TrilinosWrappers::MPI::BlockVector distributed_solution(locally_relevant_partitioning,
                                                            mpi_communicator);
    distributed_solution = solution;

    std::vector<Tensor<1, dim> > fe_solution_gradient;
    double Energy = 0.0;

    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = hp_dof_handler.begin_active(),
    endc = hp_dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        if (cell->is_locally_owned())
        {
            if(cell->material_id() == 1)
            {
                hp_fe_values.reinit(cell);
                const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();
                const unsigned int  &n_q_points = fe_values.n_quadrature_points;
                const std::vector<Point<dim> > &quadrature_points = fe_values.get_quadrature_points();

                fe_solution_gradient.resize(n_q_points);
                fe_values[phi_fe].get_function_gradients (distributed_solution, fe_solution_gradient);
                std::vector<double>    coefficient_values (n_q_points);
                function_material_coefficients.value_list (fe_values.get_quadrature_points(),
                                                           coefficient_values);

                Assert(n_q_points == fe_values.get_quadrature_points().size(), ExcInternalError());

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    // Get the x co-ord to the quadrature point
                    const double radial_distance = quadrature_points[q_point][0];
                    // If dim == 2, assembly using axisymmetric formulation
                    const double coord_transformation_scaling = ( dim == 2
                                                                  ?
                                                                    2.0 * dealii::numbers::PI * radial_distance
                                                                  :
                                                                    1.0);
                    const double mu_r_mu_0 = coefficient_values[q_point];

                    Energy += (0.5 * mu_r_mu_0 *
                               coord_transformation_scaling *
                               fe_solution_gradient[q_point].norm_square() *
                               fe_values.JxW(q_point));
                }
            }

        }
    }
    Energy = Utilities::MPI::sum (Energy, mpi_communicator);
    pcout << "Total energy: " << Energy << std::endl;
}

// To determine the true error in residual for the problem
template <int dim>
void MSP_Toroidal_Membrane<dim>::get_error_residual(Errors &error_residual)
{
    TrilinosWrappers::MPI::BlockVector error_res(locally_owned_partitioning,
                                                 mpi_communicator);
    error_res = system_rhs;
    constraints.set_zero(error_res);

    error_residual.norm = error_res.l2_norm();
    error_residual.u = error_res.block(u_block).l2_norm();
    error_residual.phi = error_res.block(phi_block).l2_norm();
}

// To determine the true error in Newton update for the problem
template <int dim>
void MSP_Toroidal_Membrane<dim>::get_error_update(const TrilinosWrappers::MPI::BlockVector &newton_update,
                                                  Errors &error_update)
{
    TrilinosWrappers::MPI::BlockVector error_ud(locally_owned_partitioning,
                                                mpi_communicator);
    error_ud = newton_update;
    constraints.set_zero(error_ud);

    error_update.norm = error_ud.l2_norm();
    error_update.u = error_ud.block(u_block).l2_norm();
    error_update.phi = error_ud.block(phi_block).l2_norm();
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

      // before starting the simulation output the grid
      output_results(cycle);
      // Declare the incremental solution update
      TrilinosWrappers::MPI::BlockVector solution_delta(locally_owned_partitioning,
                                                        locally_relevant_partitioning,
                                                        mpi_communicator);
      // Can add a loop over load domain here later
      // currently single load step taken
      solution_delta = 0.0;
      solve_nonlinear_system(solution_delta);
      // update the total solution for current load step
      solution += solution_delta;
      output_results(cycle);

      compute_error ();
//      postprocess_energy();
    }
}

//explicit instantiation for class template
template class MSP_Toroidal_Membrane<2>;
template class MSP_Toroidal_Membrane<3>;
