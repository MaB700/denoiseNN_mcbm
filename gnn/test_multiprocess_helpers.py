import sys

def get_procfs_path():
    """Return updated psutil.PROCFS_PATH constant."""
    """Copied from psutil code, and modified to fix an error."""
    return sys.modules['psutil'].PROCFS_PATH

def cpu_count_physical():
    """Return the number of physical cores in the system."""
    """Copied from psutil code, and modified to fix an error."""
    # Method #1 doesn't work for some dual socket topologies.
    # # Method #1
    # core_ids = set()
    # for path in glob.glob(
    #         "/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
    #     with open_binary(path) as f:
    #         core_ids.add(int(f.read()))
    # result = len(core_ids)
    # if result != 0:
    #     return result

    # Method #2
    physical_logical_mapping = {}
    mapping = {}
    current_info = {}
    with open('%s/cpuinfo' % get_procfs_path(), "rb") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                # print(current_info)
                # new section
                if (b'physical id' in current_info and
                        b'cpu cores' in current_info):
                    mapping[current_info[b'physical id']] = \
                        current_info[b'cpu cores']
                if (b'physical id' in current_info and
                        b'core id' in current_info and
                        b'processor' in current_info):
                    # print(current_info[b'physical id'] * 1000 + current_info[b'core id'])
                    if current_info[b'physical id'] * 1000 + current_info[b'core id'] not in physical_logical_mapping:
                        physical_logical_mapping[current_info[b'physical id'] * 1000 + current_info[b'core id']] = current_info[b'processor']
                current_info = {}
            else:
                # ongoing section
                if (line.startswith(b'physical id') or
                        line.startswith(b'cpu cores') or
                        line.startswith(b'core id') or
                        line.startswith(b'processor')):
                    key, value = line.split(b'\t:', 1)
                    current_info[key.rstrip()] = int(value.rstrip())

    physical_processor_ids = []
    for key in sorted(physical_logical_mapping.keys()):
        physical_processor_ids.append(physical_logical_mapping[key])

    result = sum(mapping.values())
    # return result or None  # mimic os.cpu_count()
    return result, physical_processor_ids