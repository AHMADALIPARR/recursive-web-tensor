//! Minimal JSON value, parser, and serializer (std-only).
//! Sufficient for the manifests, configs, and model files this CLI
//! reads and writes. Numbers are f64; integers round-trip exactly
//! below 2^53, which covers every count in the reference trainer.

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(BTreeMap<String, Value>),
}

impl Value {
    pub fn obj() -> Value {
        Value::Obj(BTreeMap::new())
    }
    pub fn put(&mut self, key: &str, val: Value) {
        if let Value::Obj(m) = self {
            m.insert(key.to_string(), val);
        }
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        if let Value::Obj(m) = self {
            m.get(key)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let Value::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }
    pub fn as_arr(&self) -> Option<&Vec<Value>> {
        if let Value::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
    pub fn s(text: &str) -> Value {
        Value::Str(text.to_string())
    }
    pub fn n(num: i64) -> Value {
        Value::Num(num as f64)
    }
    pub fn f(num: f64) -> Value {
        Value::Num(num)
    }
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub fn to_string(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Num(n) => {
            if n.fract() == 0.0 && n.is_finite() && n.abs() < 9e15 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        Value::Str(s) => format!("\"{}\"", escape(s)),
        Value::Arr(a) => {
            let parts: Vec<String> = a.iter().map(to_string).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Obj(m) => {
            let parts: Vec<String> = m
                .iter()
                .map(|(k, val)| format!("\"{}\":{}", escape(k), to_string(val)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.pos < self.bytes.len() && (self.bytes[self.pos] as char).is_whitespace() {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.pos))
        }
    }
    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') => self.lit("true", Value::Bool(true)),
            Some(b'f') => self.lit("false", Value::Bool(false)),
            Some(b'n') => self.lit("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(format!("unexpected input at byte {}", self.pos)),
        }
    }
    fn lit(&mut self, word: &str, val: Value) -> Result<Value, String> {
        if self.bytes.len() >= self.pos + word.len()
            && &self.bytes[self.pos..self.pos + word.len()] == word.as_bytes()
        {
            self.pos += word.len();
            Ok(val)
        } else {
            Err(format!("bad literal at byte {}", self.pos))
        }
    }
    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let c = self.bytes[self.pos];
            if c.is_ascii_digit() || c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "bad number encoding".to_string())?;
        text.parse::<f64>()
            .map(Value::Num)
            .map_err(|_| format!("bad number '{}'", text))
    }
    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        while let Some(c) = self.peek() {
            match c {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let e = self.peek().ok_or("unterminated escape")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'u' => {
                            let hex = self.bytes.get(self.pos..self.pos + 4)
                                .ok_or("short unicode escape")?;
                            let hex = std::str::from_utf8(hex)
                                .map_err(|_| "bad unicode escape")?;
                            let cp = u32::from_str_radix(hex, 16)
                                .map_err(|_| "bad unicode escape")?;
                            self.pos += 4;
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err("bad escape".to_string()),
                    }
                }
                _ => {
                    // Consume one UTF-8 char.
                    let rest = &self.bytes[self.pos..];
                    let s = std::str::from_utf8(rest).map_err(|_| "bad utf-8")?;
                    let ch = s.chars().next().ok_or("unterminated string")?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        Err("unterminated string".to_string())
    }
    fn array(&mut self) -> Result<Value, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
    }
    fn object(&mut self) -> Result<Value, String> {
        self.eat(b'{')?;
        let mut map = BTreeMap::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Obj(map));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.eat(b':')?;
            let val = self.value()?;
            map.insert(key, val);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Obj(map));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }
}

/// Parse a JSON document. Trailing non-whitespace is an error.
pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { bytes: text.as_bytes(), pos: 0 };
    let v = p.value()?;
    p.ws();
    if p.pos != p.bytes.len() {
        return Err("trailing input after JSON value".to_string());
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut o = Value::obj();
        o.put("name", Value::s("a\"b\nc"));
        o.put("n", Value::n(42));
        o.put("f", Value::f(0.5));
        o.put("ok", Value::Bool(true));
        let mut arr = vec![Value::n(1), Value::s("x")];
        arr.push(Value::Null);
        o.put("arr", Value::Arr(arr));
        let text = to_string(&o);
        let back = parse(&text).expect("parse");
        assert_eq!(back, o);
    }

    #[test]
    fn parse_errors() {
        assert!(parse("{bad}").is_err());
        assert!(parse("[1,]").is_err());
        assert!(parse("").is_err());
        assert!(parse("{\"a\":1} extra").is_err());
    }
}
