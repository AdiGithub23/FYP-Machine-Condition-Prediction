# services/fake_influx_reader.py

from fake_data.data_generator import generate_fake_point, generate_fake_timeseries

class FakeInfluxReader:
    """
    This class mimics real InfluxDB read operations.
    Later, when InfluxDB is ready, we replace this.
    """

    def get_latest_point(self):
        """Return the latest fake sensor reading."""
        return generate_fake_point()

    def get_fake_timeseries(self, n=360):
        """Return fake 360 data points."""
        return generate_fake_timeseries(n)
