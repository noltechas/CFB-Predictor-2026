// Build a complete single-file scorer for the documented JSON function API.
// Weights remain embedded and initialize lazily; no Storage requests at runtime.
const fs=require('fs');const path=require('path');const root=path.resolve(__dirname,'..');const esbuild=require(root+'/mobile/node_modules/esbuild');
let source=fs.readFileSync(root+'/supabase/functions/live-scorer/index.ts','utf8');
source='import { frozenWeights, frozenMeta } from "cfb:frozen-model";\n'+source;
source=source.replace('Deno.readFile(new URL("../../model/weights.bin", import.meta.url))','Promise.resolve((() => { const raw=atob(frozenWeights); const bytes=new Uint8Array(raw.length); for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i); return bytes; })())').replace('Deno.readTextFile(new URL("../../model/weights.json", import.meta.url))','Promise.resolve(frozenMeta)');
esbuild.buildSync({stdin:{contents:source,loader:'ts',resolveDir:root+'/supabase/functions/live-scorer'},bundle:true,format:'esm',platform:'neutral',external:['npm:*','cfb:frozen-model'],outfile:path.join(require('os').tmpdir(),'cfb-scorer-code.js')});
let bundled=fs.readFileSync(path.join(require('os').tmpdir(),'cfb-scorer-code.js'),'utf8');bundled=bundled.replace(/import \{ frozenWeights, frozenMeta \} from "cfb:frozen-model";/,'const frozenWeights='+JSON.stringify(fs.readFileSync(root+'/supabase/model/weights.bin').toString('base64'))+'; const frozenMeta='+JSON.stringify(fs.readFileSync(root+'/supabase/model/weights.json','utf8'))+';');
if(bundled.includes('cfb:frozen-model'))throw Error('Unresolved model');fs.writeFileSync((process.argv[2] || path.join(require('os').tmpdir(),'cfb-scorer-complete.js')),bundled);console.log('Complete scorer bundle bytes',Buffer.byteLength(bundled));
