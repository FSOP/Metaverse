def inbound_info(inbound, eop):
    return {
        "angle_x": inbound[0],
        "angle_y": inbound[1],
        "angle_z": inbound[2],
        "velocity_x": inbound[3],
        "velocity_y": inbound[4],
        "velocity_z": inbound[5],
        "EOP": eop
    }
def Orbit_struct(time_stamp, r, v):
    return {
        "time_stamp": time_stamp,
        "r1": r[0],
        "r2": r[1],
        "r3": r[2],
        "v1": v[0],
        "v2": v[1],
        "v3": v[2]
    }

def SAT_struc(sat_ID, sat_name, sat_type, RCS, orbit, weight, material):
    return {
        "SAT_ID"    : sat_ID,
        "SAT_Name"  : sat_name,
        "SAT_TYPE"  : sat_type,
        "SAT_RCS"   : RCS,
        "orbit"     : orbit,
        "weight"    : weight,
        "material"  : material
    }

def COLLI_Info(angle_r, velocity_v):
    return {
        "angle_x": angle_r[0],
        "angle_y": angle_r[1],
        "angle_z": angle_r[2],
        "velocity_x": velocity_v[0],
        "velocity_y": velocity_v[1],
        "velocity_z": velocity_v[2]
    }

def reassemble_orbit(ephemeris):
    res_ephemeris = []
    for ep in ephemeris:
        res_ephemeris.append(Orbit_struct(ep[0], ep[1:4], ep[4:7]))
    return res_ephemeris

def crash(crash_id, creation_date, start_time, SAT_info, inbound_info, orbit_crash, time_crash, point_crash, prob_crash):
    return {
        "crash_id": crash_id,
        "creation_date": creation_date,
        "start_time": start_time,
        "SAT_info": SAT_info,
        "inbound_info": inbound_info,
        "orbit_CRASH": orbit_crash,
        "time_CRASH": time_crash,
        "point_CRASH": point_crash,
        "prob_CRASH": prob_crash
    }