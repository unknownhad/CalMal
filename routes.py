from malware_detect import MalwareDetect


def add_routes_to_resource(_api):
    _api.add_resource(MalwareDetect, '/get-malware_predict', strict_slashes=False)
