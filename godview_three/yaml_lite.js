// A small YAML reader for configs/twin/godview.yaml (block mappings, flow mappings and lists, scalars, comments).
// It is not a general YAML parser: the static judge server cannot convert YAML, and godview.yaml is re-read at every
// load because the reference agent fills its gain and exposure. Block lists ("- x") and folded scalars are handled
// only in their simplest form.

const SCALARS = { null: null, '~': null, true: true, false: false, on: true, off: false, yes: true, no: false };

function scalar(s) {
  s = s.trim();
  if (s === '') return null;
  if ((s[0] === '"' && s.endsWith('"')) || (s[0] === "'" && s.endsWith("'"))) return s.slice(1, -1);
  const low = s.toLowerCase();
  if (low in SCALARS) return SCALARS[low];
  if (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return Number(s);
  return s;
}

/** Strip a trailing comment (a '#' after whitespace, outside quotes). */
function stripComment(line) {
  let q = null;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) { if (c === q) q = null; continue; }
    if (c === '"' || c === "'") q = c;
    else if (c === '#' && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i);
  }
  return line;
}

/** Parse a flow collection or scalar starting at s[i]; returns [value, nextIndex]. */
function flow(s, i = 0) {
  const ws = () => { while (i < s.length && /\s/.test(s[i])) i++; };
  ws();
  if (s[i] === '[') {
    i++; const out = [];
    for (;;) {
      ws();
      if (s[i] === ']') { i++; return [out, i]; }
      const [v, j] = flow(s, i); out.push(v); i = j; ws();
      if (s[i] === ',') i++;
    }
  }
  if (s[i] === '{') {
    i++; const out = {};
    for (;;) {
      ws();
      if (s[i] === '}') { i++; return [out, i]; }
      let k = '';
      while (i < s.length && s[i] !== ':') k += s[i++];
      i++;
      const [v, j] = flow(s, i); out[k.trim()] = v; i = j; ws();
      if (s[i] === ',') i++;
    }
  }
  let q = null, tok = '';
  while (i < s.length) {
    const c = s[i];
    if (q) { tok += c; i++; if (c === q) q = null; continue; }
    if (c === '"' || c === "'") { q = c; tok += c; i++; continue; }
    if (c === ',' || c === ']' || c === '}') break;
    tok += c; i++;
  }
  return [scalar(tok), i];
}

export function parseYaml(text) {
  const lines = text.split(/\r?\n/).map(stripComment).map((l) => l.replace(/\s+$/, '')).filter((l) => l.trim() !== '');
  const root = {};
  const stack = [{ indent: -1, obj: root }];
  for (let n = 0; n < lines.length; n++) {
    const line = lines[n];
    const indent = line.length - line.trimStart().length;
    const body = line.trim();
    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();
    const parent = stack[stack.length - 1];
    if (body.startsWith('- ')) {
      if (!Array.isArray(parent.obj)) continue;
      parent.obj.push(flow(body.slice(2))[0]);
      continue;
    }
    const c = body.indexOf(':');
    if (c < 0) continue;
    const key = body.slice(0, c).trim();
    const rest = body.slice(c + 1).trim();
    if (rest === '' || rest === '>-' || rest === '|' || rest === '>') {
      const next = lines[n + 1];
      const isList = next && next.trim().startsWith('- ');
      const child = isList ? [] : {};
      parent.obj[key] = child;
      stack.push({ indent, obj: child });
    } else {
      parent.obj[key] = flow(rest)[0];
    }
  }
  return root;
}
