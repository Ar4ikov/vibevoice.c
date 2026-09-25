/* What the Apple Neural Engine does with the shapes TurboQwen runs. */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdio.h>

static double now_ms(void) {
    return [NSDate timeIntervalSinceReferenceDate] * 1000.0;
}

static NSURL* compile(NSURL* pkg) {
    NSError* e = nil;
    NSURL* c = [MLModel compileModelAtURL:pkg error:&e];
    if (!c) { printf("  compile failed: %s\n", e.localizedDescription.UTF8String); }
    return c;
}

/* Which device Core ML picked for each op. */
static void plan(NSURL* compiled, MLComputeUnits units) {
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = units;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled configuration:cfg
                   completionHandler:^(MLComputePlan* p, NSError* err) {
        if (!p) { printf("  plan: %s\n", err.localizedDescription.UTF8String);
                  dispatch_semaphore_signal(sem); return; }
        MLModelStructure* st = p.modelStructure;
        MLModelStructureProgram* prog = st.program;
        if (!prog) { printf("  plan: not an ML program\n");
                     dispatch_semaphore_signal(sem); return; }
        int ane = 0, gpu = 0, cpu = 0, other = 0;
        MLModelStructureProgramFunction* fn = prog.functions[@"main"];
        for (MLModelStructureProgramOperation* op in fn.block.operations) {
            MLComputePlanDeviceUsage* u = [p computeDeviceUsageForMLProgramOperation:op];
            if (!u) continue;
            id<MLComputeDeviceProtocol> d = u.preferredComputeDevice;
            if ([d isKindOfClass:[MLNeuralEngineComputeDevice class]]) ane++;
            else if ([d isKindOfClass:[MLGPUComputeDevice class]]) gpu++;
            else if ([d isKindOfClass:[MLCPUComputeDevice class]]) cpu++;
            else other++;
        }
        printf("  ops: %d on the neural engine, %d on the GPU, %d on the CPU"
               "%s\n", ane, gpu, cpu, other ? ", some elsewhere" : "");
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
}

static MLMultiArray* make_input(MLModel* m, NSString* name) {
    MLFeatureDescription* d = m.modelDescription.inputDescriptionsByName[name];
    NSArray<NSNumber*>* shape = d.multiArrayConstraint.shape;
    MLMultiArrayDataType t = d.multiArrayConstraint.dataType;
    NSError* e = nil;
    MLMultiArray* a = [[MLMultiArray alloc] initWithShape:shape dataType:t error:&e];
    if (!a) { printf("  input: %s\n", e.localizedDescription.UTF8String); return nil; }
    [a getMutableBytesWithHandler:^(void* p, NSInteger n, NSArray<NSNumber*>* s) {
        (void)s;
        if (t == MLMultiArrayDataTypeFloat32) {
            float* f = (float*)p;
            for (NSInteger i = 0; i < n / 4; i++) f[i] = (float)((i % 17) - 8) * 0.05f;
        } else { memset(p, 0, (size_t)n); }
    }];
    return a;
}

static void bench(const char* label, NSURL* compiled, MLComputeUnits units,
                  int iters, double flops) {
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = units;
    NSError* e = nil;
    double t_load = now_ms();
    MLModel* m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&e];
    t_load = now_ms() - t_load;
    if (!m) { printf("  %s: load failed: %s\n", label, e.localizedDescription.UTF8String); return; }
    NSString* iname = m.modelDescription.inputDescriptionsByName.allKeys.firstObject;
    MLMultiArray* x = make_input(m, iname);
    if (!x) return;
    MLDictionaryFeatureProvider* in =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{iname: x} error:&e];
    id<MLFeatureProvider> out = [m predictionFromFeatures:in error:&e];   /* warm */
    if (!out) { printf("  %s: predict failed: %s\n", label, e.localizedDescription.UTF8String); return; }
    for (int i = 0; i < 3; i++) out = [m predictionFromFeatures:in error:&e];
    const double t0 = now_ms();
    for (int i = 0; i < iters; i++) out = [m predictionFromFeatures:in error:&e];
    const double ms = (now_ms() - t0) / iters;
    if (flops > 0)
        printf("  %-22s %8.3f ms   %6.2f GFLOP/s   (load %.0f ms)\n",
               label, ms, flops / (ms * 1e6), t_load);
    else
        printf("  %-22s %8.3f ms   (load %.0f ms)\n", label, ms, t_load);
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        if (argc < 3) { printf("usage: probe <model.mlpackage> <iters> [flops]\n"); return 2; }
        NSURL* pkg = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        const int iters = atoi(argv[2]);
        const double flops = argc > 3 ? atof(argv[3]) : 0.0;
        printf("%s\n", argv[1]);
        NSURL* c = compile(pkg);
        if (!c) return 1;
        plan(c, MLComputeUnitsAll);
        bench("all (ANE first)", c, MLComputeUnitsAll, iters, flops);
        bench("cpu + neural engine", c, MLComputeUnitsCPUAndNeuralEngine, iters, flops);
        bench("cpu + gpu", c, MLComputeUnitsCPUAndGPU, iters, flops);
        bench("cpu only", c, MLComputeUnitsCPUOnly, iters, flops);
    }
    return 0;
}
