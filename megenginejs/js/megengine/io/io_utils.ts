import { TypedArray } from "../dtypes";


export function arrayBufferToBase64String(buffer: ArrayBuffer): string {
  const buf = new Uint8Array(buffer);
  let s = '';
  for (let i = 0, l = buf.length; i < l; i++) {
    s += String.fromCharCode(buf[i]);
  }
  return btoa(s);
}

export function base64StringToArrayBuffer(str: string): ArrayBuffer {
  const s = atob(str);
  const buffer = new Uint8Array(s.length);
  for (let i = 0; i < s.length; ++i) {
    buffer.set([s.charCodeAt(i)], i);
  }
  return buffer.buffer;
}

export function concatenateTypedArrays(xs: TypedArray[]): ArrayBuffer {
  if (xs === null) {
    throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
  }

  let totalByteLength = 0;
  const normalizedXs: TypedArray[] = [];
  xs.forEach((x: TypedArray) => {
    totalByteLength += x.byteLength;
    // tslint:disable:no-any
    normalizedXs.push(
        x.byteLength === x.buffer.byteLength ? x :
                                               new (x.constructor as any)(x));
    if (!(x as any instanceof Float32Array || x as any instanceof Int32Array ||
          x as any instanceof Uint8Array)) {
      throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
    }
    // tslint:enable:no-any
  });

  const y = new Uint8Array(totalByteLength);
  let offset = 0;
  normalizedXs.forEach((x: TypedArray) => {
    y.set(new Uint8Array(x.buffer), offset);
    offset += x.byteLength;
  });

  return y.buffer;
}