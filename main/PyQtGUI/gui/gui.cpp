#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <Python.h>
#include "CPyHelper.h"
#include <cstdint>
#include <errno.h>


int main(int argc, char *argv[])
{
  CPyHelper pInstance;

 //use INSTALLED_IN sets in makefile.am to @prefix@
  std::string instPath(INSTALLED_IN);
  std::string filename = instPath + "/Script/Main.py";
  
  if(!getenv("USERDIR")){
    if (access("fit_skel_creator.py", F_OK) != -1 && access("algo_skel_creator.py", F_OK) != -1) {
        std::cout<<"\n \tUses .py files defined in the current directory, ex: fit_skel_creator.py, algo_skel_creator.py\n"<<std::endl;
	if (access("Main.py", F_OK) != -1) {
	    std::cout << "Using Main.py from current directory" << std::endl;
	    char cwd[1024];
	    if(getcwd(cwd, 1024)) {
		instPath = cwd;
		filename = instPath + "/Main.py";
	    } else {
		std::cerr << "Failed to set cwd and load Main.py from there!!! Falling back to " << instPath << std::endl;
	    }
	}
    } else{
        std::cout << "\n \tIf you want to use your .py files, ex: fit_skel_creator.py, algo_skel_creator.py"<< std::endl;
        std::cout << "\tPlease define USERDIR env variable as the directory where your .py files are located"<< std::endl;
        std::cout << "\tEx: export USERDIR=/EXEMPLE/OF/PATH"<< std::endl;
        std::cout << "\tOr have those .py files in the current working directory\n"<< std::endl;
    }
  } else {
      const char* path = getenv("USERDIR"); // we know its set
      std::string path_string = path;
      std::cout << "Adding USERDIR: " << path_string << std::endl;
      std::string testfile = path_string + "/Main.py";
      if (access(testfile.c_str(), F_OK) != -1) {
	  instPath = path_string;
	  filename = instPath + "/Main.py";
      } else {
	  std::cerr << "Failed to load Main.py from USERDIR!!! Falling back to " << instPath << std::endl;
      }
  }

  try {
    PyObject *obj = Py_BuildValue("s", filename.c_str());    
    FILE *file = _Py_fopen_obj(obj, "r");
    if(file != NULL) {
      PyRun_SimpleFile(file, filename.c_str());
    }
    else {
      std::string errmsg = "Cannot open QtPy main from: " + filename;
      throw std::invalid_argument(errmsg);
    }
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << '\n';
  }

  return 0;
}
