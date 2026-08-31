/**
 * XLSX-Brücke: SheetJS → dasselbe Zellraster, das der Kern liest.
 *
 * Bewusst getrennt, damit `importTable` ohne die xlsx-Abhängigkeit test- und
 * nutzbar bleibt. Die Bibliothek wird erst hier dynamisch geladen; sie ist der
 * größte Einzelposten im Bündel und wird nur beim Excel-Import gebraucht.
 */
import { importMarketGrid, importAssemblageGrid, detectKind, type AssemblageOptions } from "./importTable.js";

export async function xlsxToGrid(data: ArrayBuffer | Uint8Array, sheet?: string | number): Promise<string[][]> {
  const XLSX = await import("xlsx");
  const wb = XLSX.read(data, { type: "array" });
  const name = typeof sheet === "string" ? sheet : wb.SheetNames[typeof sheet === "number" ? sheet : 0];
  const ws = wb.Sheets[name];
  if (!ws) throw new Error(`Arbeitsblatt nicht gefunden: ${String(sheet ?? 0)}`);
  return XLSX.utils.sheet_to_json<string[]>(ws, { header: 1, raw: false, defval: "" }) as unknown as string[][];
}

export async function listSheets(data: ArrayBuffer | Uint8Array): Promise<string[]> {
  const XLSX = await import("xlsx");
  return XLSX.read(data, { type: "array", bookSheets: true }).SheetNames;
}

export async function importMarketXLSX(data: ArrayBuffer | Uint8Array, sheet?: string | number) {
  return importMarketGrid(await xlsxToGrid(data, sheet));
}

export async function importAssemblageXLSX(data: ArrayBuffer | Uint8Array, opts: AssemblageOptions = {}, sheet?: string | number) {
  return importAssemblageGrid(await xlsxToGrid(data, sheet), opts);
}

export async function detectKindXLSX(data: ArrayBuffer | Uint8Array, sheet?: string | number) {
  return detectKind(await xlsxToGrid(data, sheet));
}
