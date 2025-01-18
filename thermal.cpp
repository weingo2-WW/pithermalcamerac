
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tuple>
#include <cstdint>

#include <bcm2835.h>

using namespace cv;
using namespace std;

struct eeparameters {
  int16_t kVdd;
  int16_t vdd25;
  double KvPTAT;
  double KtPTAT;
  int16_t vPTAT25;
  double alphaPTAT;
  int16_t gainEE;
  double tgc;
  double KsTa;
  int16_t resolutionEE;
  int16_t calibrationModeEE;
  double ksTo[5];
  double ct[5];
  int32_t alpha[768];
  int16_t alphaScale;
  int16_t offset[768];
  int16_t kta[768];
  int16_t ktaScale;
  int16_t kv[768];
  int16_t kvScale;
  int16_t cpAlpha[2];
  int16_t cpOffset[2];
  double ilChessC[3];
  int16_t cpKta;
  int16_t cpKv;

};

const double SCALEALPHA = 0.000001;

void ExtractEEParameters ( uint16_t ee_data[832], struct eeparameters* eeparams ) {
  eeparams->kVdd = (int8_t)((ee_data[51] & 0xFF00) >> 8);
  eeparams->kVdd *= 32;
  eeparams->vdd25 = ee_data[51] & 0x00FF;
  eeparams->vdd25 = ((eeparams->vdd25 - 256) << 5) - 8192;

  eeparams->KvPTAT = (ee_data[50] & 0xFC00) >> 10;
  if (eeparams->KvPTAT > 31)
      eeparams->KvPTAT -= 64;
  eeparams->KvPTAT /= 4096;
  eeparams->KtPTAT = ee_data[50] & 0x03FF;
  if (eeparams->KtPTAT > 511)
      eeparams->KtPTAT -= 1024;
  eeparams->KtPTAT /= 8;
  eeparams->vPTAT25 = ee_data[49];
  eeparams->alphaPTAT = (ee_data[16] & 0xF000) / pow(2, 14) + 8;

  eeparams->gainEE = ee_data[48];

  eeparams->tgc = ee_data[60] & 0x00FF;
  if ( eeparams->tgc > 127 )
      eeparams->tgc -= 256;
  eeparams->tgc /= 32;

  eeparams->resolutionEE = (ee_data[56] & 0x3000) >> 12;

  eeparams->KsTa = (int8_t)((ee_data[60] & 0xFF00) >> 8);
  eeparams->KsTa /= 8192; 

  int16_t step = ((ee_data[63] & 0x3000) >> 12) * 10;
  eeparams->ct[0] = -40;
  eeparams->ct[1] = 0;
  eeparams->ct[2] = (ee_data[63] & 0x00F0) >> 4;
  eeparams->ct[3] = (ee_data[63] & 0x0F00) >> 8;
  eeparams->ct[2] *= step;
  eeparams->ct[3] = eeparams->ct[2] + eeparams->ct[3] * step;

  int64_t KsToScale = (ee_data[63] & 0x000F) + 8;
  KsToScale = 1 << KsToScale;

  eeparams->ksTo[0] = (int8_t)(ee_data[61] & 0x00FF);
  eeparams->ksTo[1] = (int8_t)((ee_data[61] & 0xFF00) >> 8);
  eeparams->ksTo[2] = (int8_t)(ee_data[62] & 0x00FF);
  eeparams->ksTo[3] = (int8_t)((ee_data[62] & 0xFF00) >> 8);

  for ( int i = 0; i < 4; i++ )
      eeparams->ksTo[i] /= KsToScale;
  eeparams->ksTo[4] = -0.0002;

  double offsetSP[2]={0,0};
  offsetSP[0] = ee_data[58] & 0x03FF;
  if (offsetSP[0] > 511)
      offsetSP[0] -= 1024;
  offsetSP[1] = (ee_data[58] & 0xFC00) >> 10;
  if (offsetSP[1] > 31)
      offsetSP[1] -= 64;
  offsetSP[1] += offsetSP[0];

  double alphaSP[2]={0,0};
  alphaSP[0] = ee_data[57] & 0x03FF;
  if (alphaSP[0] > 511)
      alphaSP[0] -= 1024;
  int16_t alphaScale = ((ee_data[32] & 0xF000) >> 12) + 27;
  alphaSP[0] /= pow(2, alphaScale);
  alphaSP[1] = (ee_data[57] & 0xFC00) >> 10;
  if (alphaSP[1] > 31)
      alphaSP[1] -= 64;
  alphaSP[1] = (1 + alphaSP[1] / 128) * alphaSP[0];

  eeparams->cpKta = (int8_t)(ee_data[59] & 0x00FF);
  int16_t ktaScale1 = ((ee_data[56] & 0x00F0) >> 4) + 8;
  eeparams->cpKta /= pow(2, ktaScale1);

  eeparams->cpKv = (int8_t)((ee_data[59] & 0xFF00) >> 8);
  int16_t kvScale = (ee_data[56] & 0x0F00) >> 8;
  eeparams->cpKv /= pow(2, kvScale);

  eeparams->cpAlpha[0] = alphaSP[0];
  eeparams->cpAlpha[1] = alphaSP[1];
  eeparams->cpOffset[0] = offsetSP[0];
  eeparams->cpOffset[1] = offsetSP[1];

  // extract alpha
  int16_t accRemScale = ee_data[32] & 0x000F;
  int16_t accColumnScale = (ee_data[32] & 0x00F0) >> 4;
  int16_t accRowScale = (ee_data[32] & 0x0F00) >> 8;
  alphaScale = ((ee_data[32] & 0xF000) >> 12) + 30;
  int16_t alphaRef = ee_data[33];
  int16_t accRow[24];
  int16_t accColumn[32];
  double alphaTemp[768];
  for ( int i = 0; i < 6; i ++ ) {
    int p = i * 4;
    accRow[p + 0] = ee_data[34 + i] & 0x000F;
    accRow[p + 1] = (ee_data[34 + i] & 0x00F0) >> 4;
    accRow[p + 2] = (ee_data[34 + i] & 0x0F00) >> 8;
    accRow[p + 3] = (ee_data[34 + i] & 0xF000) >> 12;
  }

  for ( int i = 0; i < 24; i ++ ) {
    if (accRow[i] > 7)
        accRow[i] -= 16;
  }

  for ( int i = 0; i < 8; i ++ ) {
    int p = i * 4;
    accColumn[p + 0] = ee_data[40 + i] & 0x000F;
    accColumn[p + 1] = (ee_data[40 + i] & 0x00F0) >> 4;
    accColumn[p + 2] = (ee_data[40 + i] & 0x0F00) >> 8;
    accColumn[p + 3] = (ee_data[40 + i] & 0xF000) >> 12;
  }

  for ( int i = 0; i < 32; i ++ ) {
      if (accColumn[i] > 7)
          accColumn[i] -= 16;
  }

  for ( int i = 0; i < 24; i ++ ) {
    for ( int j = 0; j < 32; j ++ ) {
       int p = 32 * i + j;
       alphaTemp[p] = (ee_data[64 + p] & 0x03F0) >> 4;
       if (alphaTemp[p] > 31)
           alphaTemp[p] -= 64;
       alphaTemp[p] *= 1 << accRemScale;
       alphaTemp[p] += (
           alphaRef
           + (accRow[i] << accRowScale)
           + (accColumn[j] << accColumnScale)
       );
       alphaTemp[p] /= pow(2, alphaScale);
       alphaTemp[p] -= eeparams->tgc * (eeparams->cpAlpha[0] + eeparams->cpAlpha[1]) / 2;
       alphaTemp[p] = SCALEALPHA / alphaTemp[p];
    }
  }
  double temp = -HUGE_VAL;
  for ( int i = 0; i < 24; i ++ )
    for ( int j = 0; j < 32; j ++ )
       temp = (alphaTemp[32 * i + j]>temp)?alphaTemp[32 * i + j]:temp;

  alphaScale = 0;
  while (temp < 32768) {
       temp *= 2;
       alphaScale += 1;
  }

  for ( int i = 0; i < 768; i++ ) {
    temp = alphaTemp[i] * pow(2, alphaScale);
    eeparams->alpha[i] = (int32_t)(temp + 0.5);
  }

  eeparams->alphaScale = alphaScale;

  int16_t occRow[24];
  int16_t occColumn[32];

  int16_t occRemScale = ee_data[16] & 0x000F;
  int16_t occColumnScale = (ee_data[16] & 0x00F0) >> 4;
  int16_t occRowScale = (ee_data[16] & 0x0F00) >> 8;
  int16_t offsetRef = ee_data[17];

  for ( int i = 0; i < 6; i ++ ) {
    int p = i * 4;
    occRow[p + 0] = ee_data[18 + i] & 0x000F;
    occRow[p + 1] = (ee_data[18 + i] & 0x00F0) >> 4;
    occRow[p + 2] = (ee_data[18 + i] & 0x0F00) >> 8;
    occRow[p + 3] = (ee_data[18 + i] & 0xF000) >> 12;
  }

  for ( int i = 0; i < 24; i ++ ) {
    if (occRow[i] > 7)
        occRow[i] -= 16;
  }

  for ( int i = 0; i < 8; i ++ ) {
    int p = i * 4;
    occColumn[p + 0] = ee_data[24 + i] & 0x000F;
    occColumn[p + 1] = (ee_data[24 + i] & 0x00F0) >> 4;
    occColumn[p + 2] = (ee_data[24 + i] & 0x0F00) >> 8;
    occColumn[p + 3] = (ee_data[24 + i] & 0xF000) >> 12;
  }

  for ( int i = 0; i < 32; i ++ ) {
      if (occColumn[i] > 7)
          occColumn[i] -= 16;
  }

  for ( int i = 0; i < 24; i ++ ) {
    for ( int j = 0; j < 32; j ++ ) {
      int p = 32 * i + j;
      eeparams->offset[p] = (ee_data[64 + p] & 0xFC00) >> 10;
      if (eeparams->offset[p] > 31)
          eeparams->offset[p] -= 64;
      eeparams->offset[p] *= 1 << occRemScale;
      eeparams->offset[p] += (
          offsetRef
          + (occRow[i] << occRowScale)
          + (occColumn[j] << occColumnScale)
      );
    }
  }

  int16_t KtaRC[4];

  int8_t KtaRoCo = (ee_data[54] & 0xFF00) >> 8;
  KtaRC[0] = KtaRoCo;

  int8_t KtaReCo = ee_data[54] & 0x00FF;
  KtaRC[2] = KtaReCo;

  int8_t KtaRoCe = (ee_data[55] & 0xFF00) >> 8;
  KtaRC[1] = KtaRoCe;

  int8_t KtaReCe = ee_data[55] & 0x00FF;
  KtaRC[3] = KtaReCe;

  ktaScale1 = ((ee_data[56] & 0x00F0) >> 4) + 8;
  int16_t ktaScale2 = ee_data[56] & 0x000F;

  double ktaTemp[768];
  for ( int i = 0; i < 24; i ++ ) {
    for ( int j = 0; j < 32; j ++ ) {
      int p = 32 * i + j;
      int16_t split = 2 * (p / 32 - (p / 64) * 2) + p % 2;
      ktaTemp[p] = (ee_data[64 + p] & 0x000E) >> 1;
      if (ktaTemp[p] > 3)
          ktaTemp[p] -= 8;
      ktaTemp[p] *= 1 << ktaScale2;
      ktaTemp[p] += KtaRC[split];
      ktaTemp[p] /= pow(2, ktaScale1);
      // ktaTemp[p] = ktaTemp[p] * mlx90640->offset[p];
    }
  }

  temp = fabs(ktaTemp[0]);
  for ( int i = 0; i < 768; i++ )
      temp = (fabs(ktaTemp[i])>temp)?fabs(ktaTemp[i]):temp;

  ktaScale1 = 0;
  while (temp < 64) {
      temp *= 2;
      ktaScale1 += 1;
  }

  for ( int i = 0; i < 768; i++ ) {
    double temp = ktaTemp[i] * pow(2, ktaScale1);
    if (temp < 0)
        eeparams->kta[i] = temp - 0.5;
    else
        eeparams->kta[i] = temp + 0.5;
  }
  eeparams->ktaScale = ktaScale1;

  int16_t KvT[4];
  double kvTemp[768];

  int8_t KvRoCo = (ee_data[52] & 0xF000) >> 12;
  if ( KvRoCo > 7 )
      KvRoCo -= 16;
  KvT[0] = KvRoCo;

  int8_t KvReCo = (ee_data[52] & 0x0F00) >> 8;
  if (KvReCo > 7)
      KvReCo -= 16;
  KvT[2] = KvReCo;

  int8_t KvRoCe = (ee_data[52] & 0x00F0) >> 4;
  if (KvRoCe > 7)
      KvRoCe -= 16;
  KvT[1] = KvRoCe;

  int8_t KvReCe = ee_data[52] & 0x000F;
  if (KvReCe > 7)
      KvReCe -= 16;
  KvT[3] = KvReCe;

  kvScale = (ee_data[56] & 0x0F00) >> 8;

  for ( int i = 0; i < 24; i ++ ) {
    for ( int j = 0; j < 32; j ++ ) {
      int p = 32 * i + j;
      int16_t split = 2 * (p / 32 - (p / 64) * 2) + p % 2;
      kvTemp[p] = KvT[split];
      kvTemp[p] /= pow(2, kvScale);
      // kvTemp[p] = kvTemp[p] * mlx90640->offset[p];
    }
  }

  temp = fabs(kvTemp[0]);
  for ( int i = 0; i < 768; i++ )
      temp = (fabs(kvTemp[i])>temp)?fabs(kvTemp[i]):temp;

  kvScale = 0;
  while (temp < 64) {
      temp *= 2;
      kvScale += 1;
  }

  for ( int i = 0; i < 768; i++ ) {
      double temp = kvTemp[i] * pow(2, kvScale);
      if (temp < 0)
          eeparams->kv[i] = temp - 0.5;
      else
          eeparams->kv[i] = temp + 0.5;
  }
  eeparams->kvScale = kvScale;

  eeparams->calibrationModeEE = (ee_data[10] & 0x0800) >> 4;
  eeparams->calibrationModeEE = eeparams->calibrationModeEE ^ 0x80;

  eeparams->ilChessC[0] = ee_data[53] & 0x003F;
  if (eeparams->ilChessC[0] > 31)
      eeparams->ilChessC[0] -= 64;
  eeparams->ilChessC[0] /= 16;

  eeparams->ilChessC[1] = (ee_data[53] & 0x07C0) >> 6;
  if (eeparams->ilChessC[1] > 15)
      eeparams->ilChessC[1] -= 32;
  eeparams->ilChessC[1] /= 2.;

  eeparams->ilChessC[2] = (ee_data[53] & 0xF800) >> 11;
  if (eeparams->ilChessC[2] > 15)
      eeparams->ilChessC[2] -= 32;
  eeparams->ilChessC[2] /= 8.0;


  /*
    def _ExtractDeviatingPixels(self) -> None:
        # pylint: disable=too-many-branches
        pixCnt = 0

        while (
            (pixCnt < 768)
            and (len(self.brokenPixels) < 5)
            and (len(self.outlierPixels) < 5)
        ):
            if eeData[pixCnt + 64] == 0:
                self.brokenPixels.append(pixCnt)
            elif (eeData[pixCnt + 64] & 0x0001) != 0:
                self.outlierPixels.append(pixCnt)
            pixCnt += 1

        if len(self.brokenPixels) > 4:
            raise RuntimeError("More than 4 broken pixels")
        if len(self.outlierPixels) > 4:
            raise RuntimeError("More than 4 outlier pixels")
        if (len(self.brokenPixels) + len(self.outlierPixels)) > 4:
            raise RuntimeError("More than 4 faulty pixels")
        # print("Found %d broken pixels, %d outliers"
        #         % (len(self.brokenPixels), len(self.outlierPixels)))

        for brokenPixel1, brokenPixel2 in self._UniqueListPairs(self.brokenPixels):
            if self._ArePixelsAdjacent(brokenPixel1, brokenPixel2):
                raise RuntimeError("Adjacent broken pixels")

        for outlierPixel1, outlierPixel2 in self._UniqueListPairs(self.outlierPixels):
            if self._ArePixelsAdjacent(outlierPixel1, outlierPixel2):
                raise RuntimeError("Adjacent outlier pixels")

        for brokenPixel in self.brokenPixels:
            for outlierPixel in self.outlierPixels:
                if self._ArePixelsAdjacent(brokenPixel, outlierPixel):
                    raise RuntimeError("Adjacent broken and outlier pixels")

    def _UniqueListPairs(self, inputList: List[int]) -> Tuple[int, int]:
        # pylint: disable=no-self-use
        for i, listValue1 in enumerate(inputList):
            for listValue2 in inputList[i + 1 :]:
                yield listValue1, listValue2

    def _ArePixelsAdjacent(self, pix1: int, pix2: int) -> bool:
        # pylint: disable=no-self-use
        pixPosDif = pix1 - pix2

        if -34 < pixPosDif < -30:
            return True
        if -2 < pixPosDif < 2:
            return True
        if 30 < pixPosDif < 34:
            return True

        return False

    def _IsPixelBad(self, pixel: int) -> bool:
        if pixel in self.brokenPixels or pixel in self.outlierPixels:
            return True

        return False
	*/

}

eeparameters GetEEParameters () {
    const size_t ee_len = 832;
    unsigned int addr=0x2400;
    uint16_t ee_data[ee_len];
    for ( int i = 0; i < ee_len; i++ ) {
      char b1 = addr, b2 = (addr>>8);
      const int len = 2;
      char cmds[2] = {b2, b1};
      char buf[len];
      uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
      uint16_t val = (buf[0]<<8)+buf[1];
      ee_data[i] = val;
      addr++;
    }
    struct eeparameters eeparams;
    ExtractEEParameters ( ee_data, &eeparams ) ;
    return eeparams;
}

double GetVdd ( uint16_t* parameter_data, struct eeparameters* eeparams ) {
  double vdd = (int16_t) parameter_data[810-32*24];

  double resolutionRAM = (parameter_data[832-32*24] & 0x0C00) >> 10;
  double resolutionCorrection = pow(2, eeparams->resolutionEE) / pow(
      2, resolutionRAM
  );
  return (resolutionCorrection * vdd - (double)eeparams->vdd25) / (double)eeparams->kVdd + 3.3;
}

double GetTa ( uint16_t* parameter_data, struct eeparameters* eeparams ) {
  double vdd = GetVdd ( parameter_data, eeparams );

  double ptat = parameter_data[800-32*24];
  double ptatArt = parameter_data[768-32*24];
  double ptatArt_d = (ptat / (ptat * eeparams->alphaPTAT + ptatArt)) * pow(2., 18);

  double ta = ptatArt_d / (1. + eeparams->KvPTAT * (vdd - 3.3)) - eeparams->vPTAT25;
  return ta / eeparams->KtPTAT + 25.;
	  
}

void ToTemperature ( uint16_t* parameter_data, struct eeparameters* eeparams, uint16_t thermal_data[32][24], double emissivity, double temperature[32][24], bool page ) {
  int subPage = parameter_data[833-32*24];

  double vdd = GetVdd ( parameter_data, eeparams ) ;
  double ta = GetTa ( parameter_data, eeparams ) ;
  const double OPENAIR_TA_SHIFT = 8.;
  double tr = ta - OPENAIR_TA_SHIFT;

  double ta4 = ta + 273.15;
  ta4 = ta4 * ta4;
  ta4 = ta4 * ta4;
  double tr4 = tr + 273.15;
  tr4 = tr4 * tr4;
  tr4 = tr4 * tr4;
  double taTr = tr4 - (tr4 - ta4) / emissivity;

  double ktaScale = pow(2, eeparams->ktaScale);
  double kvScale = pow(2, eeparams->kvScale);
  double alphaScale = pow(2, eeparams->alphaScale);

  double alphaCorrR[4];
  alphaCorrR[0] = 1. / (1 + eeparams->ksTo[0] * 40);
  alphaCorrR[1] = 1.;
  alphaCorrR[2] = 1. + eeparams->ksTo[1] * eeparams->ct[2];
  alphaCorrR[3] = alphaCorrR[2] * (1 + eeparams->ksTo[2] * (eeparams->ct[3] - eeparams->ct[2]));

  // --------- Gain calculation -----------------------------------
  double gain = (int16_t)parameter_data[778-32*24];
  gain = eeparams->gainEE / gain;

  // --------- To calculation -------------------------------------
  int16_t mode = (parameter_data[832-32*24] & 0x1000) >> 5;

  double irDataCP[2];
  irDataCP[0] = (int16_t) parameter_data[776-32*24];
  irDataCP[1] = (int16_t) parameter_data[808-32*24];
  irDataCP[0] *= gain;
  irDataCP[1] *= gain;
  irDataCP[0] -= (
      eeparams->cpOffset[0]
      * (1 + eeparams->cpKta * (ta - 25))
      * (1 + eeparams->cpKv * (vdd - 3.3))
  );
  if ( mode == eeparams->calibrationModeEE ) {
      irDataCP[1] -= (
          eeparams->cpOffset[1]
          * (1 + eeparams->cpKta * (ta - 25))
          * (1 + eeparams->cpKv * (vdd - 3.3))
      );
  } else {
      irDataCP[1] -= (
          (eeparams->cpOffset[1] + eeparams->ilChessC[0])
          * (1 + eeparams->cpKta * (ta - 25))
          * (1 + eeparams->cpKv * (vdd - 3.3))
      );
  }

  for ( int i = 0; i < 24; i ++ ) {
    for ( int j = 0; j < 32; j ++ ) {
      int p = 32 * i + j;
      // complex logic for chess pattern reading
      // even i and even j or odd i and odd j is sub page 0
      // even i and odd j or odd i and even j is sub page 1
      if ( (((!(i%2) && !(j%2))||((i%2) && (j%2))) && !page) || (((!(i%2) && (j%2))||((i%2) && !(j%2))) && page) ) {

      // if self._IsPixelBad(pixelNumber):
      //     # print("Fixing broken pixel %d" % pixelNumber)
      //     result[pixelNumber] = -273.15
      //     continue

      int ilPattern = p / 32 - (p / 64) * 2;
      int chessPattern = ilPattern ^ (p - (p / 2) * 2);
      int conversionPattern = (
          (p + 2) / 4
          - (p + 3) / 4
          + (p + 1) / 4
          - p / 4
      ) * (1 - 2 * ilPattern);

      int pattern;
      if (mode == 0)
          pattern = ilPattern;
      else
          pattern = chessPattern;

      // if ( pattern == parameter_data[833-32*24] ) 
      if ( 1 ) {
      double irData = (int16_t)thermal_data[j][i];
      irData *= gain;

      double kta = eeparams->kta[p] / ktaScale;
      double kv = eeparams->kv[p] / kvScale;
      irData -= (
          eeparams->offset[p]
          * (1. + kta * (ta - 25.))
          * (1. + kv * (vdd - 3.3))
      );

      if ( mode != eeparams->calibrationModeEE ) {
          irData += (
              eeparams->ilChessC[2] * (2 * ilPattern - 1)
              - eeparams->ilChessC[1] * conversionPattern
          );
      }

      irData = irData - eeparams->tgc * irDataCP[subPage];
      irData /= emissivity;

      double alphaCompensated = SCALEALPHA * alphaScale / (double)eeparams->alpha[p];
      alphaCompensated *= 1. + eeparams->KsTa * (ta - 25);
      // printf("Correction %e\n", alphaCompensated);

      double Sx = (
          alphaCompensated
          * alphaCompensated
          * alphaCompensated
          * (irData + alphaCompensated * taTr)
      );
      Sx = sqrt(sqrt(Sx)) * eeparams->ksTo[1];

      double To = (
          sqrt(
              sqrt(
                  irData
                  / (alphaCompensated * (1 - eeparams->ksTo[1] * 273.15) + Sx)
                  + taTr
              )
          )
          - 273.15
      );

      int torange;
      if ( To < eeparams->ct[1] )
          torange = 0;
      else if ( To < eeparams->ct[2] )
          torange = 1;
      else if ( To < eeparams->ct[3] )
          torange = 2;
      else
          torange = 3;

      To = (
          sqrt(
              sqrt(
                  irData
                  / (
                      alphaCompensated
                      * alphaCorrR[torange]
                      * (1 + eeparams->ksTo[torange] * (To - eeparams->ct[torange]))
                  )
                  + taTr
              )
          )
          - 273.15
      );

      temperature[j][i] = To*9./5.+32;
    }
    }
    }
  }
}

void GetMinMaxAvg ( double temperature[32][24], double* min, double* max, double* avg ) {
  *avg = 0;
  *max=-HUGE_VAL; *min=HUGE_VAL;
  for ( int i = 0; i < 32; i++ ) {
    for ( int j = 0; j < 24; j++ ) {
      *max = temperature[i][j]>*max?temperature[i][j]:*max;
      *min = *min>temperature[i][j]?temperature[i][j]:*min;
      *avg += temperature[i][j];
    }
  }
  *avg /= 32*24;
}
void TemperatureToColorMag ( double temperature[32][24], uint8_t color_mag[32][24] ) {
  double avg, max, min;
  GetMinMaxAvg ( temperature, &min, &max, &avg ) ;
  for ( int i = 0; i < 32; i++ ) {
    for ( int j = 0; j < 24; j++ ) {
      // color_mag[i][j] = (temperature[i][j]+40)/(300+40)*255;
      // color_mag[i][j] = (temperature[i][j]+40)/(100+40)*255;
      // color_mag[i][j] = (temperature[i][j]+10)/(30+15)*255;
      color_mag[i][j] = (temperature[i][j]-min)/(max-min)*255;
      // printf("color %d\n", color_mag[i][j]);
    }
  }
}

uint16_t GetStatusRegister() {
      const int len = 2;
      char cmds[2] = {0x80, 0x00};
      char buf[len];
      uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
      return (buf[0]<<8)+buf[1];
}

bool GetHasData() {
      uint16_t status = GetStatusRegister();
      return status&0b1000;
}

// 0 is sub-page 0
// 1 is sub-page 1
bool GetPage() {
      uint16_t status = GetStatusRegister();
      return status&0b1;
}

void ClearStatusRegister() {
      const int len = 2;
      char cmds[2] = {0x80, 0x00};
      char buf[len];
      uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
      {
        char bufw[4] = {0x80, 0x00, buf[0], buf[1]&0b1110111};
        data = bcm2835_i2c_write(bufw, 4);
      }
}

void GetThermalData(uint16_t thermal_data[32][24], bool page) {
  int addr = 0x400;
  for ( int j = 0; j < 24; j++ ) {
    for ( int i = 0; i < 32; i++ ) {
      // complex logic for chess pattern reading
      if ( (((!(i%2) && !(j%2))||((i%2) && (j%2))) && !page) || (((!(i%2) && (j%2))||((i%2) && !(j%2))) && page) ) {
        // printf("i %d j %d p %d page %d\n", i, j, p, page);
        const int len = 2;
        char b1 = addr, b2 = (addr>>8);
        char cmds[2] = {b2, b1};
        char buf[len];
        uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
        uint16_t val = (buf[0]<<8)+buf[1];
        thermal_data[i][j] = val;
      }
      addr++;
    }
  }
}

uint16_t GetControlRegister() {
      const int len = 2;
      char cmds[2] = {0x80, 0x0D};
      char buf[len];
      uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
      return (buf[0]<<8)+buf[1];
}

void SetRefreshRate( int rate ) {
  uint16_t control = GetControlRegister() ;
  uint16_t mask = 0x0380;
  uint16_t mask_value = 0;
  switch (rate) {
    case 0: 
      mask_value = 0x0000; break;
    case 1: 
      mask_value = 0x0080; break;
    case 2: 
      mask_value = 0x0100; break;
    case 4: 
      mask_value = 0x0180; break;
    case 8: 
      mask_value = 0x0200; break;
    case 16: 
      mask_value = 0x0280; break;
    case 32: 
      mask_value = 0x0300; break;
    case 64: 
      mask_value = 0x0380; break;
  }
  uint16_t control_masked = (control&(!mask))|mask_value;
  control_masked |= 1<<12; // chess pattern
  control_masked |= (1<<11)+(1<<10); // 19-bit ADC
  control_masked &= ~(1<<3); // Sub-page toggle on
  control_masked &= ~(1<<2); // No data hold
  control_masked |= 1<<0; // Sub-pate mode on
  // printf("control_masked 0x%x\n", control_masked);
  char b1 = control_masked>>8;
  char b2 = control_masked|0x0380;
  char bufw[4] = {0x80, 0x0D, b1, b2};
  uint8_t data = bcm2835_i2c_write(bufw, 4);
}

bool IsPowerOfTwo ( int x ) {
  return ( x & (x-1) ) == 0;
}

void GetParameterData(uint16_t parameter_data[66]) {
  int addr = 0x400+32*24;
  for ( int i = 0; i < 66; i++ ) {
    char b1 = addr, b2 = (addr>>8);
    const int len = 2;
    char cmds[2] = {b2, b1};
    char buf[len];
    uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
    uint16_t val = (buf[0]<<8)+buf[1];
    parameter_data[i] = val;
    addr++;
  }
  { // control register
    const int len = 2;
    char cmds[2] = {0x80, 0x0D};
    char buf[len];
    uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
    parameter_data[832-32*24] = (buf[0]<<8)+buf[1];
  }
  { // status register
    const int len = 2;
    char cmds[2] = {0x80, 0x00};
    char buf[len];
    uint8_t data = bcm2835_i2c_write_read_rs(cmds, len, buf, len);
    parameter_data[833-32*24] = buf[1] & 0x0001;
  }
}

FILE *pipeout = NULL;
void cleanup(int dummy) {
    bcm2835_i2c_end();   
    bcm2835_close();
    printf("... done!\n");
    if (pipeout) pclose(pipeout);
    exit(0);
}

static volatile int keepRunning = 1;
void intHandler(int dummy) {
    keepRunning = 0;
}


int main(int argc, char *argv[])
{
    bool stream = true;
    bool video = false;
    bool grey_scale = false;
    unsigned int baud_rate = 200000;
    unsigned int refresh_rate = 4;
    int opt;
    bool onlyrefresh = false;
    while ((opt = getopt(argc, argv, "fnvgb:r:h")) != -1) {
        switch (opt) {
        case 'f': 
		printf("only computing ideal refresh rate and exiting\n") ; 
		onlyrefresh = true; break;
        case 'v': 
		printf("Streaming to a video using ffmpeg\n") ; 
		video = true; break;
        case 'n': 
		printf("disabling python web stream server\n") ; 
		stream = false; break;
        case 'g': 
		printf("Using grey scale\n") ; 
		grey_scale = true; break;
        case 'r': 
		  sscanf(argv[optind-1], "%d", &refresh_rate) ; 
		  printf("new refresh rate: %d\n", refresh_rate) ; 
		  if ( !IsPowerOfTwo ( refresh_rate ) || refresh_rate>64 ) {
	            fprintf(stderr, "Error: bad refresh rate\n");
		    return 1;
		  } 

		  break;
        case 'b': 
		  sscanf(argv[optind-1], "%d", &baud_rate) ; 
		  printf("new baud rate: %d\n", baud_rate) ; 
		  if ( baud_rate > 1000000 || baud_rate < 1000 ) {
	            fprintf(stderr, "Error: bad baud rate\n");
		    return 1;
		  } else if ( baud_rate > 200000 ) 
	            printf("warning: bad baud rate above 200000 did not work well on my pi zero W\n");

		  // printf("baud rate: %s\n", argv[optind-1]) ; 
		  // printf("baud rate: %c\n", optarg) ; 
		  break;
        // case '?':  
        //      printf("unknown option: %c\n", optopt); 
        case 'h':
        default:
            fprintf(stderr, "Usage: sudo %s [-hgvnf] [-b baud_rate] [-r refresh_rate]\n", argv[0]);
            fprintf(stderr, "\t-g uses greyscale instead of the jet colorscale\n");
            fprintf(stderr, "\t-b changes the default baud rate of 200000 to something else\n");
            fprintf(stderr, "\t-r changes the default refresh rate of 4 Hz. Values are 0, 2, 4, 6, 8, 16, 32, 64.\n");
            fprintf(stderr, "\t-v outputs to a video using ffmpeg\n");
            fprintf(stderr, "\t-n disables python fileserver for web stream\n");
            fprintf(stderr, "\t-f computes ideal refresh rate and exits\n");
            fprintf(stderr, "\t-h displays the usage message and exits\n");
            exit(EXIT_FAILURE);
        }
    }
    printf("start!\n");
    // i2c boilerplate
    if (!bcm2835_init()) return 1;
    // I2C begin if specified    
    bcm2835_i2c_begin();
    // signal(SIGINT, cleanup);
    signal(SIGINT, intHandler);


    bcm2835_i2c_setSlaveAddress(0x33);
    bcm2835_i2c_set_baudrate ( baud_rate ); 
    SetRefreshRate( refresh_rate ) ;
    if ( onlyrefresh ) {
      time_t time0 = time(NULL);
      int frame_count = 0;
      while(keepRunning) {
        bool hasdata = GetHasData();
        while ( !hasdata ) {
          hasdata = GetHasData();
          usleep(1000);
          time_t now = time(NULL);
          if ( (now-time0) > 60 ) {
            fprintf(stderr, "Error timeout after 30 seconds of no data\n");
            cleanup(0);        
            return 1;
          }
        }
        ClearStatusRegister();
        frame_count++;
        time_t now = time(NULL);
        if ( (now-time0) > 30 )  break;
      }
      time_t now = time(NULL);
      printf("Ideal %f fps", (double)frame_count/(double)(now-time0));
      cleanup(0);        
      return 0;
    }

    int height=240+30;
    int width=320;
    int fps = 3;
    if ( video ) {
      // Use a "generic" example (write the output video in output.mkv video file).
      // ffmpeg -y -f rawvideo -r 10 -video_size 320x240 -pixel_format bgr24 -i pipe: -vcodec libx264 -crf 24 -pix_fmt yuv420p output.mkv
      std::string ffmpeg_cmd = std::string("ffmpeg -y -f rawvideo -r ") + std::to_string(fps) +
                               " -video_size " + std::to_string(width) + "x" + std::to_string(height) +
                               " -pixel_format bgr24 -i pipe: -vcodec libx264 -crf 24 -pix_fmt yuv420p output.mkv";
      //https://batchloaf.wordpress.com/2017/02/12/a-simple-way-to-read-and-write-audio-and-video-files-in-c-using-ffmpeg-part-2-video/
      pipeout = popen(ffmpeg_cmd.c_str(), "w");     //Linux (assume ffmpeg exist in /usr/bin/ffmpeg (and in path).
    }

    if ( stream ) {
      pid_t pid = fork();
      if (pid < 0) {
          perror("fork failed");
          exit(1);
      } else if (pid == 0) {
        system("python3 -m http.server 2> /dev/null");
	fprintf(stderr, "Error: Stream unexpectedly closed\n");
	exit(1);
      } 
      sleep(1); // let the server start
    }

    eeparameters eeparams = GetEEParameters ();

    time_t time0 = time(NULL);
    int frame_count = 0;
    uint16_t thermal_data[32][24];
    double temperature[32][24];
    uint8_t color_mag[32][24];
    while(keepRunning) {
      bool hasdata = GetHasData();
      while ( !hasdata ) {
        hasdata = GetHasData();
        usleep(1000);
        time_t now = time(NULL);
	if ( (now-time0) > 30 ) {
	  fprintf(stderr, "Error timeout after 30 seconds of no data\n");
          cleanup(0);        
          return 1;
	}
      }
      ClearStatusRegister();

      bool page = GetPage() ;
      GetThermalData(thermal_data, page) ;
           
      uint16_t parameter_data[66];
      GetParameterData(parameter_data) ;
      ToTemperature ( parameter_data, &eeparams, thermal_data, 0.95, temperature, page ) ;

      // Construct from and array
      TemperatureToColorMag ( temperature, color_mag ) ;
      double avg, max, min;
      GetMinMaxAvg ( temperature, &min, &max, &avg ) ;
      char text[80];
      sprintf(text, "Min %.0fF Max %.0fF Avg %.0fF", min, max, avg);

      cv::Mat img = cv::Mat(32, 24, CV_8U, &color_mag);
      cv::transpose(img,img);

      cv::Mat img_tmp;
      // 1 is horizontal, 0 is vertical, -1 is both
      cv::flip(img, img_tmp, 1);
      img = img_tmp;

      cv::resize(img, img, Size(), 10, 10, INTER_CUBIC);
      if ( !grey_scale ) {
        cv::applyColorMap(img, img, COLORMAP_JET);
      } else {
        Point minLoc, maxLoc;
        double min, max;
        minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
        cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
        // image, point, color, type, size, thickness, line_type
        cv::drawMarker ( img, maxLoc, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10, 2, 8);
        cv::drawMarker ( img, minLoc, cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 10, 2, 8);
      }
      // copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
      copyMakeBorder( img, img, 30, 0, 0, 0, BORDER_CONSTANT, cv::Scalar(0, 0, 0) );
      cv::putText(img, text, cv::Point(1, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
      cv::imwrite("out.jpg", img);

      time_t now = time(NULL);
      frame_count++;
      if ( now-time0 ) {
        printf("\r            \r%d fps", frame_count);
        fflush(stdout);
        frame_count=0;
        time0=now;
      }

      if ( video ) {
        //Write width*height*3 bytes to stdin pipe of FFmpeg sub-process (assume frame data is continuous in the RAM).
        fwrite(img.data, 1, width*height*3, pipeout);
        fflush(pipeout);
      }
    }

    // std::string greyArrWindow = "Grey Array Image";
    // cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    // cv::imshow(greyArrWindow, greyImg);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // This I2C end is done after a transfer if specified

    cleanup(0);        
    return 0;
}
