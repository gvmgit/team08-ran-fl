# Windows patches to the Crosstrainer utility (local copy, not part of the service)

The utility we were given (`crosstrainer-utility`, from the organisers' distribution bucket) is written for Linux: it launches the
CDK through a shell and copies folders with `cp -rL`. To run it from a Windows laptop we changed the installed copy in the places
below. The service, the training client and the model code are not touched. The original files were kept as `*.orig`; the
`fc /n` output below shows the original first and the patched text second, for each changed block.

## dist\cli.js

```text
***** original
  152:              }
  153:              execFileSync(resolveCdkBin(), ["deploy", ...cdkArgs], { stdio: "inherit", cwd: PROJECT_ROOT });
  154:              return;
***** patched
  152:              }
  153:              execFileSync(...(process.platform === "win32" ? [process.execPath, [resolveCdkBin(), "deploy", ...cdkArgs]] : [reso
  154:  lveCdkBin(), ["deploy", ...cdkArgs]]), { stdio: "inherit", cwd: PROJECT_ROOT });
  155:              return;
*****

```

## dist\scripts\create-membership.js

```text
***** original
  369:      if (!existsSync(modelDest)) {
  370:          execFileSync("cp", ["-rL", modelCodeDir, modelDest], { stdio: "pipe" });
  371:      }
***** patched
  369:      if (!existsSync(modelDest)) {
  370:          process.getBuiltinModule("node:fs").cpSync(modelCodeDir, modelDest, { recursive: true, dereference: true });
  371:      }
*****

***** original
  727:              if (!existsSync(buildModelDir)) {
  728:                  execFileSync("cp", ["-rL", modelCodeDir, buildModelDir], { stdio: "pipe" });
  729:              }
***** patched
  727:              if (!existsSync(buildModelDir)) {
  728:                  process.getBuiltinModule("node:fs").cpSync(modelCodeDir, buildModelDir, { recursive: true, dereference: true })
  729:  ;
  730:              }
*****

```

## dist\scripts\start-training.js

```text
***** original
  756:          if (!existsSync(modelDest)) {
  757:              execFileSync("cp", ["-rL", modelCodeDir, modelDest], { stdio: "pipe" });
  758:          }
***** patched
  756:          if (!existsSync(modelDest)) {
  757:              process.getBuiltinModule("node:fs").cpSync(modelCodeDir, modelDest, { recursive: true, dereference: true });
  758:          }
*****

```

## dist\scripts\train.js

```text
***** original
   86:      const cdkArgs = ["-c", `persona=${persona}`, ...args.flatMap((a) => ["-c", a])];
   87:      execFileSync(resolveCdkBin(), ["deploy", ...cdkArgs], { stdio: "inherit", cwd: PROJECT_ROOT });
   88:  }
***** patched
   86:      const cdkArgs = ["-c", `persona=${persona}`, ...args.flatMap((a) => ["-c", a])];
   87:      execFileSync(...(process.platform === "win32" ? [process.execPath, [resolveCdkBin(), "deploy", ...cdkArgs]] : [resolveCdkBi
   88:  n(), ["deploy", ...cdkArgs]]), { stdio: "inherit", cwd: PROJECT_ROOT });
   89:  }
*****

```
