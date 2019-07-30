

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


class BeamRunner():
    """
    This should be pulling in data from a yaml config - which in
    turn may be getting populated with environment variables
    from a local secrets repository.
    :return:
    """
    options = PipelineOptions()
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = 'my-project-id'
    google_cloud_options.job_name = 'myjob'
    google_cloud_options.staging_location = 'gs://your-bucket-name-here/staging'
    google_cloud_options.temp_location = 'gs://your-bucket-name-here/temp'
    options.view_as(StandardOptions).runner = 'DataflowRunner'

    def __init__(self):
        #read yaml
        self.options
