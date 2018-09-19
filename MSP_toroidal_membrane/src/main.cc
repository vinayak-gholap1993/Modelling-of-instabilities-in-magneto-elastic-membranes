#include <MSP_toroidal_membrane.h>

using namespace dealii;

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
//        pcout << "Running with " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
//              << " MPI processes" << std::endl;
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
