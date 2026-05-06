from dataclasses import dataclass, asdict
from typing import Any, List, Tuple
from datetime import datetime


@dataclass
class OrbitStruct:
    time_stamp: Any
    r: Tuple[float, float, float]
    v: Tuple[float, float, float]

    def to_dict(self):
        return {
            "time_stamp": self.time_stamp,
            "r1": self.r[0],
            "r2": self.r[1],
            "r3": self.r[2],
            "v1": self.v[0],
            "v2": self.v[1],
            "v3": self.v[2],
        }


@dataclass
class SAT:
    SAT_ID: Any
    SAT_Name: str
    SAT_TYPE: str
    SAT_RCS: Any
    orbit: Any
    weight: Any
    material: Any

    def to_dict(self):
        return {
            "SAT_ID": self.SAT_ID,
            "SAT_Name": self.SAT_Name,
            "SAT_TYPE": self.SAT_TYPE,
            "SAT_RCS": self.SAT_RCS,
            "orbit": self.orbit,
            "weight": self.weight,
            "material": self.material,
        }


@dataclass
class COLLIInfo:
    angle_x: float
    angle_y: float
    angle_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float

    def to_dict(self):
        return asdict(self)


def inbound_info(inbound, eop):
    # keep legacy dict API
    return {
        "angle_x": inbound[0],
        "angle_y": inbound[1],
        "angle_z": inbound[2],
        "velocity_x": inbound[3],
        "velocity_y": inbound[4],
        "velocity_z": inbound[5],
        "EOP": eop,
    }


def Orbit_struct(time_stamp, r, v):
    # compatibility wrapper returning same dict shape as before
    return OrbitStruct(time_stamp, tuple(r), tuple(v)).to_dict()


def SAT_struc(sat_ID, sat_name, sat_type, RCS, orbit, weight, material):
    # compatibility wrapper returning same dict shape as before
    s = SAT(sat_ID, sat_name, sat_type, RCS, orbit, weight, material)
    return s.to_dict()


def COLLI_Info(angle_r, velocity_v):
    c = COLLIInfo(angle_r[0], angle_r[1], angle_r[2], velocity_v[0], velocity_v[1], velocity_v[2])
    return c.to_dict()


def reassemble_orbit(ephemeris: List[Any]):
    res_ephemeris = []
    for ep in ephemeris:
        # ep is expected to be a sequence: [time, r0,r1,r2, v0,v1,v2]
        try:
            time_stamp = ep[0]
            r = tuple(ep[1:4])
            v = tuple(ep[4:7])
        except Exception:
            # fallback for unexpected shapes
            continue
        res_ephemeris.append(OrbitStruct(time_stamp, r, v).to_dict())
    return res_ephemeris


def crash(crash_id, creation_date, start_time, SAT_info, inbound_info_obj, orbit_crash, time_crash, point_crash, prob_crash):
    return {
        "crash_id": crash_id,
        "creation_date": creation_date,
        "start_time": start_time,
        "SAT_info": SAT_info,
        "inbound_info": inbound_info_obj,
        "orbit_CRASH": orbit_crash,
        "time_CRASH": time_crash,
        "point_CRASH": point_crash,
        "prob_CRASH": prob_crash,
    }