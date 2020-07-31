
#include "precision.h"
#include "VarDriver3D.h"
#include "Xml.h"
#include <iostream>
#include "timing/gptl.h"

#ifdef PETSC
#include "petscsys.h" 
#endif

int main (int argc, char *argv[]) {
  // Initialize our timers:
  GPTLinitialize();
  GPTLstart("Total");

#ifdef PETSC
  // PETSc Initialize
   auto ierr = PetscInitialize(&argc, &argv, 0, 0);
   if (ierr) return ierr;
#endif 

	// Set up our XML document:
	XMLDocument xml;

  // Read the command line argument to get the XML configuration file
  if (argc >=2) {
    GPTLstart("Main::Init");
		XMLError ec = xml.LoadFile(argv[1]);
		if (ec != XML_SUCCESS) {
			std::cout << "Error opening XML file : " << argv[1] << std::endl;
			return 1;
		}

		// Get the root element of the XML file ([bdobbins] - we can clean this up a lot)
  	XMLElement* root = xml.FirstChildElement("samurai");

    // Generic driver which will be instanced by the configuration specification
    VarDriver *driver = new VarDriver3D();

    // Make sure we were able to create a driver, then drive
    if (driver != NULL) {
      // Do the analysis
      if(!driver->initialize(*root)) {
				delete driver;
				return EXIT_FAILURE;
      }
      GPTLstop("Main::Init");

      GPTLstart("Main::Run");
			if(!driver->run()) {
				delete driver;
				return EXIT_FAILURE;
      }
      GPTLstop("Main::Run");

      GPTLstart("Main::Finalize");
			if(!driver->finalize()) {
				//BUG To avoid an free allocation error when running 
				//BUG with __OPENACC__ comment out the following line
				// delete driver;
				return EXIT_FAILURE;
      }
      GPTLstop("Main::Finalize");
                        //BUG to avoid a free allocation error when running comment out the following line
			//BUG kludge
		        // delete driver;
      std::cout << "Analysis successful!\n";
      GPTLstop("Total");
      GPTLpr(0);
      //GPTLpr_summary(0);
      GPTLfinalize();
#ifdef PETSC
      ierr = PetscFinalize ();
#endif
      return EXIT_SUCCESS;
    } else {
      std:: cout << "No run mode found!" << std::endl;
      return EXIT_FAILURE;
    }
  } else {
    std::cout << "Usage: samurai <samurai_configuration.xml>\n";
    return EXIT_SUCCESS;
  }
#ifdef PETSC
  ierr = PetscFinalize ();
#endif
}
