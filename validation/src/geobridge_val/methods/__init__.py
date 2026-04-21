from . import correlation, geobridge, grnboost, velorama

REGISTRY = {
    "correlation": correlation.run,
    "geobridge":   geobridge.run,
    "grnboost":    grnboost.run,
    "velorama":    velorama.run,
}
