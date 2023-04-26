from collections import defaultdict
import os
import torch
import errno
class Analysis_Util:
    @staticmethod
    def mkdir_if_missing(dirname):
        """Create dirname if it is missing."""
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise



    @staticmethod
    def save_results(content, output_dir,fname):
        """Writes to a json file."""
        Analysis_Util.mkdir_if_missing(output_dir)
        filename = os.path.join(output_dir,fname)
        torch.save(content,filename)





