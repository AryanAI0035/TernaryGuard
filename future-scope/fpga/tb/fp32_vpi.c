/* Local simulation arithmetic oracle only; never linked into synthesis. */
#include <vpi_user.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24,"binary32 required");
static PLI_INT32 compute(PLI_BYTE8 *unused) {
    (void)unused;
    vpiHandle call=vpi_handle(vpiSysTfCall,0), args=vpi_iterate(vpiArgument,call);
    uint32_t raw[3];
    for (int i=0;i<3;i++) {
        s_vpi_value v={.format=vpiIntVal};
        vpi_get_value(vpi_scan(args),&v); raw[i]=(uint32_t)v.value.integer;
    }
    float a,b,r;
    memcpy(&a,raw+1,4);memcpy(&b,raw+2,4);
    switch(raw[0]) {case 0:r=a+b;break;case 1:r=a*b;break;
        case 2:r=a/b;break;default:r=sqrtf(a);break;}
    uint32_t bits;memcpy(&bits,&r,4);
    s_vpi_value value={.format=vpiIntVal,.value.integer=(PLI_INT32)bits};
    vpi_put_value(call,&value,0,vpiNoDelay); return 0;
}
static PLI_INT32 size(PLI_BYTE8 *unused) {(void)unused;return 32;}
static void install(void) {
    s_vpi_systf_data tf={.type=vpiSysFunc,.sysfunctype=vpiSizedFunc,
        .tfname="$tg_f32",.calltf=compute,.sizetf=size};vpi_register_systf(&tf);
}
void (*vlog_startup_routines[])(void)={install,0};
