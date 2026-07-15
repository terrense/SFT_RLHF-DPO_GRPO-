import numpy as np
_p = {'long':np.int_,'ulong':np.uint,'longlong':np.int64,'ulonglong':np.uint64,
      'int':int,'uint':np.uint,'short':np.int16,'ushort':np.uint16,'byte':np.int8,'ubyte':np.uint8,
      'float':float,'double':np.double,'bool':bool,'object':object,'str':str,'unicode':np.str_,'complex':complex}
for n,v in _p.items():
    if not hasattr(np,n):
        try: setattr(np,n,v)
        except Exception: pass
import sys
from data_juicer.tools.process_data import main
sys.exit(main())
