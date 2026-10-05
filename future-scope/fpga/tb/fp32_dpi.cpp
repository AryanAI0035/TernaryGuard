// Compiled local simulation only. AMD IP validation remains a separate gate.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
static_assert(sizeof(float)==4 && FLT_MANT_DIG==24,"binary32 required");
extern "C" uint32_t tg_f32(uint32_t op,uint32_t aa,uint32_t bb) {
    float a,b,r;
    std::memcpy(&a,&aa,4);std::memcpy(&b,&bb,4);
    switch(op) {case 0:r=a+b;break;case 1:r=a*b;break;
        case 2:r=a/b;break;default:r=std::sqrt(a);break;}
    uint32_t bits;std::memcpy(&bits,&r,4);return bits;
}
