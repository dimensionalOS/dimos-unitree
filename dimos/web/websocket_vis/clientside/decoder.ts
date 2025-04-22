import { Costmap, EncodedSomething, Grid, Vector } from "./types.ts";

export function decode(data: EncodedSomething) {
  console.log("decoding", data);
  if (data.type == "costmap") {
    return Costmap.decode(data);
  }
  if (data.type == "vector") {
    return Vector.decode(data);
  }
  if (data.type == "grid") {
    return Grid.decode(data);
  }

  return "UNKNOWN";
}
