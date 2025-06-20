from collections import deque
from typing import Deque, List
from enum import Enum


class Buffer:
    def __init__(self, buffersize: int):
        self.buffersize = buffersize
        self.buffer: Deque[int] = deque(maxlen=buffersize)
        self.filled = False

    def fill(self) -> bool:
        return self.filled or len(self.buffer) == self.buffersize

    def add(self, x: int):
        self.buffer.append(x)
        if len(self.buffer) == self.buffersize:
            self.filled = True

    def get_count(self) -> int:
        return len(self.buffer)

    def get_avg(self) -> int:
        if len(self.buffer) == 0:
            return 0
        return int(sum(self.buffer) / len(self.buffer))

    def get_start(self) -> int:
        if len(self.buffer) == 0:
            return 0
        return self.buffer[0]


class MorphologyFilter:
    def __init__(self):
        self.buffer = Buffer(8)
        self.previous_output = 0

    def compute_hpf(self, data: int) -> int:
        """
        Morphology-based high-pass filter for spike removal.
        This implements a simple morphological opening operation.
        """
        self.buffer.add(data)

        if not self.buffer.fill():
            # Return input until buffer is filled
            return data

        # Simple morphological filter: median of the buffer
        sorted_values = sorted(list(self.buffer.buffer))
        median_idx = len(sorted_values) // 2
        median_value = sorted_values[median_idx]

        # High-pass: input minus low-frequency component (median)
        output = data - median_value

        # Apply some smoothing to reduce artifacts
        smoothed_output = int(0.7 * output + 0.3 * self.previous_output)
        self.previous_output = smoothed_output

        return smoothed_output


class NotchEcgFilter:
    """
    Notch filter for removing power line interference at specific frequencies.
    Supports multiple frequencies: 50Hz, 60Hz, 100Hz, 120Hz
    """

    _AR_NOTCH_50 = [
        0.003191240254179,
        0.0004023940264151,
        -0.0001885801538513,
        -0.0009946831852538,
        -0.001685560730961,
        -0.001925539236658,
        -0.001522098520199,
        -0.0005333680269218,
        0.0007194709980928,
        0.001753659840532,
        0.00211053956438,
        0.001555161495633,
        0.0002054196275165,
        -0.001473234044321,
        -0.002821356225562,
        -0.00323826184536,
        -0.002437785956292,
        -0.0006020639639701,
        0.001642440731701,
        0.003433224139613,
        0.00399917612592,
        0.0029834511109,
        0.0006320284688226,
        -0.002244537303345,
        -0.004541846384,
        -0.005277350761264,
        -0.00399854327205,
        -0.001016226022768,
        0.002646048467218,
        0.005591790415681,
        0.006574141535607,
        0.005011559714335,
        0.001288145357135,
        -0.003306923789776,
        -0.007013360286963,
        -0.008263641499684,
        -0.006327439949779,
        -0.00168691496025,
        0.004027508514384,
        0.008619314005463,
        0.01013386384461,
        0.007694575089546,
        0.00194836645966,
        -0.005058222198752,
        -0.01059125763832,
        -0.01228684033168,
        -0.009149802199265,
        -0.002075927422513,
        0.006360133784057,
        0.01282697530772,
        0.01454636270676,
        0.01047862169169,
        0.001880829233369,
        -0.008043633154571,
        -0.01532864770893,
        -0.01682902712556,
        -0.01157460778684,
        -0.001309588720338,
        0.01005658143724,
        0.01793648057692,
        0.01891927953391,
        0.01224722226935,
        0.0002723463538699,
        -0.01235701830664,
        -0.02050441759353,
        -0.02064434756012,
        -0.01239113494028,
        0.001207636123782,
        0.01479084265898,
        0.02280902055994,
        0.0218076650344,
        0.01192150131267,
        -0.003068404398047,
        -0.01718331659017,
        -0.02464989957766,
        -0.02228358651129,
        -0.01085052334453,
        0.005159304009261,
        0.01931250078256,
        0.02583690561678,
        0.02199847336905,
        0.009256256293575,
        -0.007293906013231,
        -0.02097527018786,
        0.9737515356737,
        -0.02097527018786,
        -0.007293906013231,
        0.009256256293575,
        0.02199847336905,
        0.02583690561678,
        0.01931250078256,
        0.005159304009261,
        -0.01085052334453,
        -0.02228358651129,
        -0.02464989957766,
        -0.01718331659017,
        -0.003068404398047,
        0.01192150131267,
        0.0218076650344,
        0.02280902055994,
        0.01479084265898,
        0.001207636123782,
        -0.01239113494028,
        -0.02064434756012,
        -0.02050441759353,
        -0.01235701830664,
        0.0002723463538699,
        0.01224722226935,
        0.01891927953391,
        0.01793648057692,
        0.01005658143724,
        -0.001309588720338,
        -0.01157460778684,
        -0.01682902712556,
        -0.01532864770893,
        -0.008043633154571,
        0.001880829233369,
        0.01047862169169,
        0.01454636270676,
        0.01282697530772,
        0.006360133784057,
        -0.002075927422513,
        -0.009149802199265,
        -0.01228684033168,
        -0.01059125763832,
        -0.005058222198752,
        0.00194836645966,
        0.007694575089546,
        0.01013386384461,
        0.008619314005463,
        0.004027508514384,
        -0.00168691496025,
        -0.006327439949779,
        -0.008263641499684,
        -0.007013360286963,
        -0.003306923789776,
        0.001288145357135,
        0.005011559714335,
        0.006574141535607,
        0.005591790415681,
        0.002646048467218,
        -0.001016226022768,
        -0.00399854327205,
        -0.005277350761264,
        -0.004541846384,
        -0.002244537303345,
        0.0006320284688226,
        0.0029834511109,
        0.00399917612592,
        0.003433224139613,
        0.001642440731701,
        -0.0006020639639701,
        -0.002437785956292,
        -0.00323826184536,
        -0.002821356225562,
        -0.001473234044321,
        0.0002054196275165,
        0.001555161495633,
        0.00211053956438,
        0.001753659840532,
        0.0007194709980928,
        -0.0005333680269218,
        -0.001522098520199,
        -0.001925539236658,
        -0.001685560730961,
        -0.0009946831852538,
        -0.0001885801538513,
        0.0004023940264151,
        0.003191240254179,
    ]
    _AR_NOTCH_60 = [
        -0.003938724014889,
        0.01498794223805,
        -0.001865117008976,
        -0.00481863345601,
        -0.003755969174496,
        -0.001422540633262,
        0.00105619593972,
        0.002764623266707,
        0.003002161360111,
        0.001673669433141,
        -0.0005227149950744,
        -0.002404336779334,
        -0.00294945158108,
        -0.001877699968029,
        0.000176607267872,
        0.002042365178714,
        0.002724130491097,
        0.001947288802137,
        0.0002579197074577,
        -0.00137424966707,
        -0.002167955417139,
        -0.0018941995205,
        -0.0008965043651263,
        0.0002789236947357,
        0.001209127755516,
        0.001732619687145,
        0.001803132181831,
        0.001324207637989,
        0.0001842960506553,
        -0.001476763571776,
        -0.003026923786416,
        -0.003466049687554,
        -0.002013053775461,
        0.001152039978604,
        0.004587232335073,
        0.006140822298526,
        0.004249085774038,
        -0.0007859392674331,
        -0.00648204619663,
        -0.009310052976893,
        -0.006838346152016,
        0.0004058293862064,
        0.008679756232888,
        0.01289067561247,
        0.009691380444807,
        -4.475403619511e-005,
        -0.0111288961872,
        -0.01676734664891,
        -0.01270105351999,
        -0.0002699991783147,
        0.01374824256746,
        0.020790367708,
        0.01574051838221,
        0.0005122066648212,
        -0.01643691913222,
        -0.02479361281234,
        -0.01867258024593,
        -0.0006655454164888,
        0.01907992865478,
        0.02859243778939,
        0.02135871608215,
        0.0007223609402317,
        -0.02154898366857,
        -0.03200446841931,
        -0.0236756570993,
        -0.0006871430328865,
        0.02372299821387,
        0.03486484988663,
        0.0255145163661,
        0.0005538391863152,
        -0.02548189953121,
        -0.03702940387603,
        -0.02676012645098,
        -0.0003603634726058,
        0.02672190470438,
        0.0383647570084,
        0.02740252623502,
        0.0001218184935861,
        -0.02738308211151,
        0.9611882288515,
        -0.02738308211151,
        0.0001218184935861,
        0.02740252623502,
        0.0383647570084,
        0.02672190470438,
        -0.0003603634726058,
        -0.02676012645098,
        -0.03702940387603,
        -0.02548189953121,
        0.0005538391863152,
        0.0255145163661,
        0.03486484988663,
        0.02372299821387,
        -0.0006871430328865,
        -0.0236756570993,
        -0.03200446841931,
        -0.02154898366857,
        0.0007223609402317,
        0.02135871608215,
        0.02859243778939,
        0.01907992865478,
        -0.0006655454164888,
        -0.01867258024593,
        -0.02479361281234,
        -0.01643691913222,
        0.0005122066648212,
        0.01574051838221,
        0.020790367708,
        0.01374824256746,
        -0.0002699991783147,
        -0.01270105351999,
        -0.01676734664891,
        -0.0111288961872,
        -4.475403619511e-005,
        0.009691380444807,
        0.01289067561247,
        0.008679756232888,
        0.0004058293862064,
        -0.006838346152016,
        -0.009310052976893,
        -0.00648204619663,
        -0.0007859392674331,
        0.004249085774038,
        0.006140822298526,
        0.004587232335073,
        0.001152039978604,
        -0.002013053775461,
        -0.003466049687554,
        -0.003026923786416,
        -0.001476763571776,
        0.0001842960506553,
        0.001324207637989,
        0.001803132181831,
        0.001732619687145,
        0.001209127755516,
        0.0002789236947357,
        -0.0008965043651263,
        -0.0018941995205,
        -0.002167955417139,
        -0.00137424966707,
        0.0002579197074577,
        0.001947288802137,
        0.002724130491097,
        0.002042365178714,
        0.000176607267872,
        -0.001877699968029,
        -0.00294945158108,
        -0.002404336779334,
        -0.0005227149950744,
        0.001673669433141,
        0.003002161360111,
        0.002764623266707,
        0.00105619593972,
        -0.001422540633262,
        -0.003755969174496,
        -0.00481863345601,
        -0.001865117008976,
        0.01498794223805,
        -0.003938724014889,
    ]
    _AR_NOTCH_100 = [
        -0.002835341706497,
        0.001357021092775,
        0.00175356890369,
        0.001129854492479,
        -3.898195586601e-005,
        -0.0002815145954316,
        0.0006559317770359,
        0.001258212258998,
        0.0004074218041138,
        -0.0008328389981926,
        -0.0006417811196973,
        0.0007845116098433,
        0.001306407285182,
        -3.890762717271e-005,
        -0.001393103820573,
        -0.0006562340289416,
        0.001256224591407,
        0.001481411305598,
        -0.0005515915152469,
        -0.001955420949703,
        -0.0004655602629604,
        0.001954569984419,
        0.001653704771924,
        -0.001245703006958,
        -0.002576224140045,
        -8.562413210186e-005,
        0.002855166175721,
        0.001770032672193,
        -0.002167032894025,
        -0.003264818645718,
        0.0004984262094065,
        0.003954047952307,
        0.001800678687775,
        -0.003339050117764,
        -0.004009382444471,
        0.001305421482753,
        0.005242021640411,
        0.001719862606388,
        -0.004761209243408,
        -0.004791344206009,
        0.002339460920238,
        0.006701631671367,
        0.00151348217106,
        -0.00642307242477,
        -0.005585993065435,
        0.00359937869174,
        0.008296651177509,
        0.001170563293345,
        -0.008293615203173,
        -0.006362934202574,
        0.005064382060221,
        0.009985481419449,
        0.0006825066860324,
        -0.01032537575322,
        -0.007089450639715,
        0.006703478948926,
        0.01171492774282,
        5.558896525801e-005,
        -0.01245684990799,
        -0.007734407708285,
        0.008469693012766,
        0.01342120413383,
        -0.0006986546027121,
        -0.01461575165985,
        -0.008264285525365,
        0.01030648212637,
        0.01503709209646,
        -0.001563704220335,
        -0.01672315382957,
        -0.008653838334245,
        0.01214742259776,
        0.01649583992593,
        -0.002509423785324,
        -0.0186922937057,
        -0.008876538768586,
        0.01392236088611,
        0.01773102537182,
        -0.003506350047652,
        -0.02044106624852,
        -0.008918181649677,
        0.0155592567876,
        0.0186858191538,
        -0.004515813569513,
        -0.02189212963195,
        -0.008769502924452,
        0.01698702848407,
        0.01931209412077,
        -0.005499668729845,
        -0.02298006790919,
        -0.00843068702697,
        0.01814655294461,
        0.01957955016412,
        -0.006416572034826,
        -0.02365340678162,
        -0.007911821111184,
        0.01898584200756,
        0.01946985961327,
        -0.007232448489726,
        0.9761181181575,
        -0.007232448489726,
        0.01946985961327,
        0.01898584200756,
        -0.007911821111184,
        -0.02365340678162,
        -0.006416572034826,
        0.01957955016412,
        0.01814655294461,
        -0.00843068702697,
        -0.02298006790919,
        -0.005499668729845,
        0.01931209412077,
        0.01698702848407,
        -0.008769502924452,
        -0.02189212963195,
        -0.004515813569513,
        0.0186858191538,
        0.0155592567876,
        -0.008918181649677,
        -0.02044106624852,
        -0.003506350047652,
        0.01773102537182,
        0.01392236088611,
        -0.008876538768586,
        -0.0186922937057,
        -0.002509423785324,
        0.01649583992593,
        0.01214742259776,
        -0.008653838334245,
        -0.01672315382957,
        -0.001563704220335,
        0.01503709209646,
        0.01030648212637,
        -0.008264285525365,
        -0.01461575165985,
        -0.0006986546027121,
        0.01342120413383,
        0.008469693012766,
        -0.007734407708285,
        -0.01245684990799,
        5.558896525801e-005,
        0.01171492774282,
        0.006703478948926,
        -0.007089450639715,
        -0.01032537575322,
        0.0006825066860324,
        0.009985481419449,
        0.005064382060221,
        -0.006362934202574,
        -0.008293615203173,
        0.001170563293345,
        0.008296651177509,
        0.00359937869174,
        -0.005585993065435,
        -0.00642307242477,
        0.00151348217106,
        0.006701631671367,
        0.002339460920238,
        -0.004791344206009,
        -0.004761209243408,
        0.001719862606388,
        0.005242021640411,
        0.001305421482753,
        -0.004009382444471,
        -0.003339050117764,
        0.001800678687775,
        0.003954047952307,
        0.0004984262094065,
        -0.003264818645718,
        -0.002167032894025,
        0.001770032672193,
        0.002855166175721,
        -8.562413210186e-005,
        -0.002576224140045,
        -0.001245703006958,
        0.001653704771924,
        0.001954569984419,
        -0.0004655602629604,
        -0.001955420949703,
        -0.0005515915152469,
        0.001481411305598,
        0.001256224591407,
        -0.0006562340289416,
        -0.001393103820573,
        -3.890762717271e-005,
        0.001306407285182,
        0.0007845116098433,
        -0.0006417811196973,
        -0.0008328389981926,
        0.0004074218041138,
        0.001258212258998,
        0.0006559317770359,
        -0.0002815145954316,
        -3.898195586601e-005,
        0.001129854492479,
        0.00175356890369,
        0.001357021092775,
        -0.002835341706497,
    ]
    _AR_NOTCH_120 = [
        -0.0007077329552539,
        0.0004374227402429,
        -0.002513043990322,
        0.001782921359836,
        0.003312258522623,
        0.0003646721323143,
        -0.002062395589157,
        0.0002605447002675,
        0.003261226815259,
        0.0008157219122604,
        -0.003351840853863,
        -0.0012076808048,
        0.004039203014024,
        0.00227207014827,
        -0.004251847915865,
        -0.003253514779951,
        0.004494591952332,
        0.004549190635791,
        -0.0044016767273,
        -0.005894351656157,
        0.004096516276108,
        0.007368375546061,
        -0.003441402912894,
        -0.008840576371352,
        0.002458687451915,
        0.01027728949366,
        -0.001102014596509,
        -0.01157774077039,
        -0.0006122707555235,
        0.01266562195632,
        0.002671814555023,
        -0.01344918208618,
        -0.00503082228635,
        0.01384952153264,
        0.007633337432553,
        -0.01379216229332,
        -0.01039494105926,
        0.01322205483581,
        0.01322093285513,
        -0.01209709021646,
        -0.01599806042988,
        0.01039745441436,
        0.0186103161347,
        -0.008134614471262,
        -0.020936368209,
        0.005343114825425,
        0.02285266871436,
        -0.002082142661501,
        -0.02425779398635,
        -0.001556591988254,
        0.02505217441054,
        0.005466695309395,
        -0.02516728586531,
        -0.009519347835828,
        0.02455536488146,
        0.01357178029544,
        -0.02319770976173,
        -0.01747863298909,
        0.02110980313091,
        0.02108957686829,
        -0.01833521110965,
        -0.02426544418598,
        0.01495227121498,
        0.02688007,
        -0.01106237266721,
        -0.02882780419668,
        0.006794852858838,
        0.03002907156542,
        -0.002291228549343,
        0.9695648129037,
        -0.002291228549343,
        0.03002907156542,
        0.006794852858838,
        -0.02882780419668,
        -0.01106237266721,
        0.02688007,
        0.01495227121498,
        -0.02426544418598,
        -0.01833521110965,
        0.02108957686829,
        0.02110980313091,
        -0.01747863298909,
        -0.02319770976173,
        0.01357178029544,
        0.02455536488146,
        -0.009519347835828,
        -0.02516728586531,
        0.005466695309395,
        0.02505217441054,
        -0.001556591988254,
        -0.02425779398635,
        -0.002082142661501,
        0.02285266871436,
        0.005343114825425,
        -0.020936368209,
        -0.008134614471262,
        0.0186103161347,
        0.01039745441436,
        -0.01599806042988,
        -0.01209709021646,
        0.01322093285513,
        0.01322205483581,
        -0.01039494105926,
        -0.01379216229332,
        0.007633337432553,
        0.01384952153264,
        -0.00503082228635,
        -0.01344918208618,
        0.002671814555023,
        0.01266562195632,
        -0.0006122707555235,
        -0.01157774077039,
        -0.001102014596509,
        0.01027728949366,
        0.002458687451915,
        -0.008840576371352,
        -0.003441402912894,
        0.007368375546061,
        0.004096516276108,
        -0.005894351656157,
        -0.0044016767273,
        0.004549190635791,
        0.004494591952332,
        -0.003253514779951,
        -0.004251847915865,
        0.00227207014827,
        0.004039203014024,
        -0.0012076808048,
        -0.003351840853863,
        0.0008157219122604,
        0.003261226815259,
        0.0002605447002675,
        -0.002062395589157,
        0.0003646721323143,
        0.003312258522623,
        0.001782921359836,
        -0.002513043990322,
        0.0004374227402429,
        -0.0007077329552539,
    ]

    def __init__(self, freq: int):
        mapping = {
            50: self._AR_NOTCH_50,
            60: self._AR_NOTCH_60,
            100: self._AR_NOTCH_100,
            120: self._AR_NOTCH_120,
        }
        self.ar_notch: List[float] = mapping.get(freq, []).copy()
        self.max_notch = len(self.ar_notch)
        self.ar_result = [0.0] * self.max_notch
        self.indx = 0

    def get_new_val(self, val: float) -> float:
        if self.max_notch == 0:
            return val

        self.ar_result[self.indx] = val
        acc = 0.0
        idx = self.indx
        for coef in self.ar_notch:
            acc += self.ar_result[idx] * coef
            idx -= 1
            if idx < 0:
                idx = self.max_notch - 1

        self.indx = (self.indx + 1) % self.max_notch
        return acc


class HPFilterType(Enum):
    HP005 = "HP005"  # 0.05 Hz cutoff
    HP015 = "HP015"  # 0.15 Hz cutoff
    HP05 = "HP05"  # 0.5 Hz cutoff


class FilterConfig:
    """Configuration class for comprehensive ECG filtering"""

    def __init__(self):
        # High-pass filter settings
        self.enable_hpf = True
        self.hpf_type = HPFilterType.HP015

        # Notch filter settings
        self.enable_notch = True
        self.notch_frequencies = [60]  # Default to 60Hz for US

        # Morphology filter settings
        self.enable_morphology = True
        self.spike_removal = True

        # Additional filter settings
        self.enable_baseline_correction = False
        self.enable_smoothing = False
        self.smoothing_window = 5


class MultiNotchFilter:
    """
    Enhanced notch filter that can handle multiple frequencies simultaneously
    """

    def __init__(self, frequencies: List[int]):
        self.filters = []
        for freq in frequencies:
            if freq in [50, 60, 100, 120]:
                self.filters.append(NotchEcgFilter(freq))

    def get_new_val(self, val: float) -> float:
        """Apply all notch filters in sequence"""
        result = val
        for notch_filter in self.filters:
            result = notch_filter.get_new_val(result)
        return result


class BaselineFilter:
    """
    Simple baseline drift correction filter using high-pass characteristics
    """

    def __init__(self, cutoff_hz: float = 0.5, sampling_rate: int = 500):
        self.cutoff = cutoff_hz
        self.fs = sampling_rate
        # Simple first-order high-pass filter
        self.rc = 1.0 / (2 * 3.14159 * cutoff_hz)
        self.dt = 1.0 / sampling_rate
        self.alpha = self.rc / (self.rc + self.dt)
        self.prev_input = 0.0
        self.prev_output = 0.0

    def get_new_val(self, val: float) -> float:
        """Apply baseline correction"""
        output = self.alpha * (self.prev_output + val - self.prev_input)
        self.prev_input = val
        self.prev_output = output
        return output


class SmoothingFilter:
    """
    Simple moving average filter for signal smoothing
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer: Deque[float] = deque(maxlen=window_size)

    def get_new_val(self, val: float) -> float:
        """Apply smoothing"""
        self.buffer.append(val)
        return sum(self.buffer) / len(self.buffer)


class HiPassFilter:
    def __init__(self, filter_type: HPFilterType):
        if filter_type == HPFilterType.HP05:
            self.HP0 = -0.9878018507
            self.HP1 = 1.9877269954
            self.GAIN = 1.006155446
        elif filter_type == HPFilterType.HP015:
            self.HP0 = -0.9963349287
            self.HP1 = 1.9963282000
            self.GAIN = 1.001837588
        elif filter_type == HPFilterType.HP005:
            self.HP0 = -0.9987734371
            self.HP1 = 1.9987726844
            self.GAIN = 1.00061384
        else:
            raise ValueError(f"Unsupported HP filter type: {filter_type}")

        self.NPOLES = 2
        # state arrays length NPOLES+1
        self.xv = [0.0] * (self.NPOLES + 1)
        self.yv = [0.0] * (self.NPOLES + 1)

    def clear_values(self):
        for i in range(self.NPOLES + 1):
            self.xv[i] = 0.0
            self.yv[i] = 0.0

    def insert_new_val(self, val: float):
        self.xv[0] = self.xv[1]
        self.xv[1] = self.xv[2]
        self.xv[2] = val / self.GAIN

    def get_new_val(self, val: float) -> float:
        self.insert_new_val(val)
        self.yv[0] = self.yv[1]
        self.yv[1] = self.yv[2]
        self.yv[2] = (
            (self.xv[0] + self.xv[2])
            - 2 * self.xv[1]
            + (self.HP0 * self.yv[0])
            + (self.HP1 * self.yv[1])
        )
        return self.yv[2]
