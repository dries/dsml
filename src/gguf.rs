use byteorder::ReadBytesExt;
use std::collections::HashMap;
use std::io::{self, Read, Seek};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
}

#[allow(non_camel_case_types)]
#[repr(u32)]
pub enum MetadataValueType {
    // The value is a 8-bit unsigned integer.
    UINT8 = 0,
    // The value is a 8-bit signed integer.
    INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    //
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    FLOAT64 = 12,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata_kv: HashMap<String, MetadataValue>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub type_: GGMLType,
    pub offset: u64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct File {
    pub header: Header,
    pub tensor_infos: Vec<TensorInfo>,
    pub padding: u64,
    pub tensor_data_begin: u64,
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

fn get_alignment(metadata_kv: &HashMap<String, MetadataValue>) -> u64 {
    metadata_kv.get("general.alignment").map_or(1, |v| {
        if let MetadataValue::Uint64(v) = v {
            *v
        } else {
            1
        }
    })
}

fn parse_string<T: Seek + Read>(stream: &mut T) -> io::Result<String> {
    let len = stream.read_u64::<byteorder::LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).unwrap())
}

fn parse_metadata_value<T: Seek + Read>(
    stream: &mut T,
    metadata_type: Option<u32>,
) -> io::Result<MetadataValue> {
    let metadata_type = match metadata_type {
        None => stream.read_u32::<byteorder::LittleEndian>()?,
        Some(t) => t,
    };
    match metadata_type {
        v if v == MetadataValueType::UINT8 as u32 => Ok(MetadataValue::Uint8(stream.read_u8()?)),
        v if v == MetadataValueType::INT8 as u32 => Ok(MetadataValue::Int8(stream.read_i8()?)),
        v if v == MetadataValueType::UINT16 as u32 => Ok(MetadataValue::Uint16(
            stream.read_u16::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::INT16 as u32 => Ok(MetadataValue::Int16(
            stream.read_i16::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::UINT32 as u32 => Ok(MetadataValue::Uint32(
            stream.read_u32::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::INT32 as u32 => Ok(MetadataValue::Int32(
            stream.read_i32::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::FLOAT32 as u32 => Ok(MetadataValue::Float32(
            stream.read_f32::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::BOOL as u32 => Ok(MetadataValue::Bool(stream.read_u8()? != 0)),
        v if v == MetadataValueType::STRING as u32 => {
            Ok(MetadataValue::String(parse_string(stream)?))
        }
        v if v == MetadataValueType::ARRAY as u32 => {
            let metadata_type = stream.read_u32::<byteorder::LittleEndian>()?;
            let count = stream.read_u64::<byteorder::LittleEndian>()?;
            let mut array = Vec::new();
            for _ in 0..count {
                array.push(parse_metadata_value(stream, Some(metadata_type))?);
            }
            Ok(MetadataValue::Array(array))
        }
        v if v == MetadataValueType::UINT64 as u32 => Ok(MetadataValue::Uint64(
            stream.read_u64::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::INT64 as u32 => Ok(MetadataValue::Int64(
            stream.read_i64::<byteorder::LittleEndian>()?,
        )),
        v if v == MetadataValueType::FLOAT64 as u32 => Ok(MetadataValue::Float64(
            stream.read_f64::<byteorder::LittleEndian>()?,
        )),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid metadata type",
        )),
    }
}

fn parse_metadata_kv<T: Seek + Read>(
    stream: &mut T,
    count: u64,
) -> io::Result<HashMap<String, MetadataValue>> {
    let mut metadata_kv = HashMap::<String, MetadataValue>::new();
    for _ in 0..count {
        let key = parse_string(stream)?;
        let value = parse_metadata_value(stream, None)?;
        metadata_kv.insert(key, value);
    }
    Ok(metadata_kv)
}

fn parse_header<T: Seek + Read>(stream: &mut T) -> io::Result<Header> {
    let mut magic = [0u8; 4];
    stream.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number",
        ));
    }
    let version = stream.read_u32::<byteorder::LittleEndian>()?;
    let tensor_count = stream.read_u64::<byteorder::LittleEndian>()?;
    let metadata_kv_count = stream.read_u64::<byteorder::LittleEndian>()?;
    let metadata_kv = parse_metadata_kv(stream, metadata_kv_count)?;
    Ok(Header {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
        metadata_kv,
    })
}

fn parse_tensor_info<T: Seek + Read>(stream: &mut T) -> io::Result<TensorInfo> {
    let name = parse_string(stream)?;
    let n_dimensions = stream.read_u32::<byteorder::LittleEndian>()?;
    let mut dimensions = Vec::new();
    for _ in 0..n_dimensions {
        dimensions.push(stream.read_u64::<byteorder::LittleEndian>()?);
    }
    let type_ = match stream.read_u32::<byteorder::LittleEndian>()? {
        v if v == GGMLType::F32 as u32 => GGMLType::F32,
        v if v == GGMLType::F16 as u32 => GGMLType::F16,
        v if v == GGMLType::Q4_0 as u32 => GGMLType::Q4_0,
        v if v == GGMLType::Q4_1 as u32 => GGMLType::Q4_1,
        v if v == GGMLType::Q5_0 as u32 => GGMLType::Q5_0,
        v if v == GGMLType::Q5_1 as u32 => GGMLType::Q5_1,
        v if v == GGMLType::Q8_0 as u32 => GGMLType::Q8_0,
        v if v == GGMLType::Q8_1 as u32 => GGMLType::Q8_1,
        v if v == GGMLType::Q2_K as u32 => GGMLType::Q2_K,
        v if v == GGMLType::Q3_K as u32 => GGMLType::Q3_K,
        v if v == GGMLType::Q4_K as u32 => GGMLType::Q4_K,
        v if v == GGMLType::Q5_K as u32 => GGMLType::Q5_K,
        v if v == GGMLType::Q6_K as u32 => GGMLType::Q6_K,
        v if v == GGMLType::Q8_K as u32 => GGMLType::Q8_K,
        v if v == GGMLType::IQ2_XXS as u32 => GGMLType::IQ2_XXS,
        v if v == GGMLType::IQ2_XS as u32 => GGMLType::IQ2_XS,
        v if v == GGMLType::IQ3_XXS as u32 => GGMLType::IQ3_XXS,
        v if v == GGMLType::IQ1_S as u32 => GGMLType::IQ1_S,
        v if v == GGMLType::IQ4_NL as u32 => GGMLType::IQ4_NL,
        v if v == GGMLType::IQ3_S as u32 => GGMLType::IQ3_S,
        v if v == GGMLType::IQ2_S as u32 => GGMLType::IQ2_S,
        v if v == GGMLType::IQ4_XS as u32 => GGMLType::IQ4_XS,
        v if v == GGMLType::I8 as u32 => GGMLType::I8,
        v if v == GGMLType::I16 as u32 => GGMLType::I16,
        v if v == GGMLType::I32 as u32 => GGMLType::I32,
        v if v == GGMLType::I64 as u32 => GGMLType::I64,
        v if v == GGMLType::F64 as u32 => GGMLType::F64,
        v if v == GGMLType::IQ1_M as u32 => GGMLType::IQ1_M,
        v if v == GGMLType::BF16 as u32 => GGMLType::BF16,
        v if v == GGMLType::TQ1_0 as u32 => GGMLType::TQ1_0,
        v if v == GGMLType::TQ2_0 as u32 => GGMLType::TQ2_0,
        _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid type")),
    };
    let offset = stream.read_u64::<byteorder::LittleEndian>()?;
    Ok(TensorInfo {
        name,
        n_dimensions,
        dimensions,
        type_,
        offset,
    })
}

fn parse_tensor_infos<T: Seek + Read>(
    stream: &mut T,
    tensor_count: u64,
) -> io::Result<Vec<TensorInfo>> {
    let mut tensor_infos = Vec::new();
    for _ in 0..tensor_count {
        tensor_infos.push(parse_tensor_info(stream)?);
    }
    Ok(tensor_infos)
}

pub fn parse_gguf<T: Seek + Read>(stream: &mut T) -> io::Result<File> {
    let header = parse_header(stream)?;
    let tensor_infos = parse_tensor_infos(stream, header.tensor_count)?;
    let position = stream.stream_position()?;
    let alignment = get_alignment(&header.metadata_kv);
    let padding = align_offset(position, alignment) - position;
    let tensor_data_begin = position + padding;

    Ok(File {
        header,
        tensor_infos,
        padding,
        tensor_data_begin,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_string() {
        let data = b"\x0d\x00\x00\x00\x00\x00\x00\x00Hello, world!";
        let string = parse_string(&mut Cursor::new(data)).unwrap();
        assert_eq!(&string, "Hello, world!");
    }

    #[test]
    fn test_parse_metadata_value() {
        let data = b"\x00\x00\x00\x00\x2a";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Uint8(42));

        let data = b"\x01\x00\x00\x00\xff";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Int8(-1));

        let data = b"\x02\x00\x00\x00\x22\xff";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Uint16(0xff22));

        let data = b"\x03\x00\x00\x00\xff\xff";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Int16(-1));

        let data = b"\x04\x00\x00\x00\x22\x33\x44\x55";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Uint32(0x55443322));

        let data = b"\x05\x00\x00\x00\xff\xff\xff\xff";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Int32(-1));

        let data = b"\x06\x00\x00\x00\x00\x00\x80\x3f";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Float32(1.0));

        let data = b"\x07\x00\x00\x00\x01";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::Bool(true));

        let data = b"\x08\x00\x00\x00\x0d\x00\x00\x00\x00\x00\x00\x00Hello, world!";
        let data = parse_metadata_value(&mut Cursor::new(data), None).unwrap();
        assert_eq!(data, MetadataValue::String("Hello, world!".to_string()));
    }
}
