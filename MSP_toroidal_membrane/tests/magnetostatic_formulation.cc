#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include "../src/MSP_toroidal_membrane.cc"

using namespace dealii;

void check()
{
    const std::string input_file (SOURCE_DIR "/magnetostatic_formulation.prm");
    ConditionalOStream pcout (std::cout,
                              (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

    // 2D (axisymmetric test)
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
//    pcout << std::endl << std::endl;

    // 3D test
//    {
//    pcout << "Running with " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
//          << " MPI processes" << std::endl;
//    const std::string title = "Running in 3-d...";
//    const std::string divider (title.size(), '=');

//    pcout
//      << divider << std::endl
//      << title << std::endl
//      << divider << std::endl;

//    MSP_Toroidal_Membrane<3> msp_toroidal_membrane (input_file);
//    msp_toroidal_membrane.run ();
//    }
}

int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    check();

    return 0;
}
