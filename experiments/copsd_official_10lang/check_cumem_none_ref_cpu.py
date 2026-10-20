"""Test reference ownership with a tiny native fixture, without importing CUDA."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
from cumem_none_ref_compat import compensate_borrowed_none


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    with tempfile.TemporaryDirectory(prefix='copsd_none_ref_') as directory:
        d=Path(directory);source=d/'fixture.c';binary=d/('none_fixture'+sysconfig.get_config_var('EXT_SUFFIX'))
        source.write_text('''#include <Python.h>
static PyObject* borrowed(PyObject* s, PyObject* a) {return Py_None;}
static PyObject* fail(PyObject* s, PyObject* a) {PyErr_SetString(PyExc_ValueError,"fixture");return NULL;}
static PyMethodDef methods[]={{"borrowed",borrowed,METH_NOARGS,NULL},{"fail",fail,METH_NOARGS,NULL},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"none_fixture",NULL,-1,methods};
PyMODINIT_FUNC PyInit_none_fixture(void){return PyModule_Create(&module);}
''')
        subprocess.run(['cc','-shared','-fPIC','-I'+sysconfig.get_path('include'),str(source),'-o',str(binary)],check=True)
        spec=importlib.util.spec_from_file_location('none_fixture',binary);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        start=sys.getrefcount(None)
        for _ in range(5):module.borrowed()
        raw_delta=sys.getrefcount(None)-start
        assert raw_delta==-5,raw_delta
        repaired=compensate_borrowed_none(module.borrowed)
        start=sys.getrefcount(None)
        for _ in range(10000):repaired()
        fixed_delta=sys.getrefcount(None)-start
        assert fixed_delta==0,fixed_delta
        raised=False
        try:compensate_borrowed_none(module.fail)()
        except ValueError:raised=True
        assert raised
    out={'scope':'Native CPU borrowed-None fixture only; not actual vLLM CUDA allocation validation.',
         'raw_five_calls_None_reference_delta':raw_delta,'repaired_10000_calls_None_reference_delta':fixed_delta,
         'native_exception_propagates':True,'GPU_used':False}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
