from .data_IO import FileReader
from .data_IO import ResultsExport, ResultsBatch

read_properties = FileReader.read_properties
read_data = FileReader.read

del FileReader
