#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <Python.h>
#include "CPyHelper.h"
#include <cstdint>
#include <errno.h>

#if defined(_WIN64) || defined(_WIN32)
#include <io.h>
#include <corecrt_io.h>
#define INSTALLED_IN "C:\\CutiePie"
#define F_OK 0
#define access(a,b) _access(a,b)
#endif

int main(int argc, char *argv[])
{
  CPyHelper pInstance;

  if(!getenv("USERDIR")){
    if (access("fit_skel_creator.py", F_OK) != -1 && access("algo_skel_creator.py", F_OK) != -1) {
        std::cout<<"\n \tUses .py files defined in the current directory, ex: fit_skel_creator.py, algo_skel_creator.py\n"<<std::endl;
    } else{
        std::cout << "\n \tIf you want to use your .py files, ex: fit_skel_creator.py, algo_skel_creator.py"<< std::endl;
        std::cout << "\tPlease define USERDIR env variable as the directory where your .py files are located"<< std::endl;
        std::cout << "\tEx: export USERDIR=/EXEMPLE/OF/PATH"<< std::endl;
        std::cout << "\tOr have those .py files in the current working directory\n"<< std::endl;

    }
  } else {
    std::cout << "All good!" << std::endl;
  }

  //use INSTALLED_IN sets in makefile.am to @prefix@
  std::string instPath(INSTALLED_IN);
  std::string filename = instPath + "/Script/Main.py";

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

