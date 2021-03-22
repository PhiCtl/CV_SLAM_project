// Auto-generated. Do not edit!

// (in-package cv_ros.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class ObjectPos {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.centroid = null;
      this.plane_vector = null;
    }
    else {
      if (initObj.hasOwnProperty('centroid')) {
        this.centroid = initObj.centroid
      }
      else {
        this.centroid = new Array(3).fill(0);
      }
      if (initObj.hasOwnProperty('plane_vector')) {
        this.plane_vector = initObj.plane_vector
      }
      else {
        this.plane_vector = new Array(3).fill(0);
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ObjectPos
    // Check that the constant length array field [centroid] has the right length
    if (obj.centroid.length !== 3) {
      throw new Error('Unable to serialize array field centroid - length must be 3')
    }
    // Serialize message field [centroid]
    bufferOffset = _arraySerializer.float64(obj.centroid, buffer, bufferOffset, 3);
    // Check that the constant length array field [plane_vector] has the right length
    if (obj.plane_vector.length !== 3) {
      throw new Error('Unable to serialize array field plane_vector - length must be 3')
    }
    // Serialize message field [plane_vector]
    bufferOffset = _arraySerializer.float64(obj.plane_vector, buffer, bufferOffset, 3);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ObjectPos
    let len;
    let data = new ObjectPos(null);
    // Deserialize message field [centroid]
    data.centroid = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    // Deserialize message field [plane_vector]
    data.plane_vector = _arrayDeserializer.float64(buffer, bufferOffset, 3)
    return data;
  }

  static getMessageSize(object) {
    return 48;
  }

  static datatype() {
    // Returns string type for a message object
    return 'cv_ros/ObjectPos';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c5326c21eb2d7acd7c7195e7a2cc19f0';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64[3] centroid 
    float64[3] plane_vector
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ObjectPos(null);
    if (msg.centroid !== undefined) {
      resolved.centroid = msg.centroid;
    }
    else {
      resolved.centroid = new Array(3).fill(0)
    }

    if (msg.plane_vector !== undefined) {
      resolved.plane_vector = msg.plane_vector;
    }
    else {
      resolved.plane_vector = new Array(3).fill(0)
    }

    return resolved;
    }
};

module.exports = ObjectPos;
