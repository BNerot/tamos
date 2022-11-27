from .data_IO import ResultsExport, ResultsBatch
from .data_IO import FileReader

read_properties = FileReader.read_properties
read_data = FileReader.read

del FileReader
