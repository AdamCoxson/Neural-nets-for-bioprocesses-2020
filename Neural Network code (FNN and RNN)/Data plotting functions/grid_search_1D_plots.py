
"""
Created on Wed Aug 19 21:51:30 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
"Finding the Optimal Optimiser"
Module: grid_search_1D_plots
Dependancies: None

This script is used to plot data obtain during the "fine-tuning" stage of my
internship, weeks 8 and 9. The data has jsut been C&P'd over from spyders 
variable explorer. The plots are the results of 1D grid search showing how neural network accuracy
evolves with each of the hyperparameters. Plotting these confirmed that the 
Bayesian optimistion script had already obtained the optimal solution across
the different hyperparameters (see week 8 slides in my 'journal' ppt file).

These have jsut been left here for completeness so you can see how training and
testing errors vary. Yes this is bad practice and full of hard code, I'm just being lazy.
"""

import matplotlib.pyplot as plt
import numpy as np


EPOCHS = np.linspace(0,1000,101)
EPOCHS[0] = 1
mse_data = [0.2092105593074411, 0.060733559416831395, 0.03393781664950972, 0.03243663859332277, 0.03177046081763274, 0.032069492950539145, 0.03149163919907321, 0.03133926582087918, 0.030289821413801748, 0.029451123465299923, 0.029147420765007, 0.02865052905209355, 0.028371926627797285, 0.02650209977259117, 0.02784548460639898, 0.027674597958937747, 0.025771528655255643, 0.026781794406329513, 0.02727384510492382, 0.02645735081220591, 0.024514068360357355, 0.028102758630452286, 0.027358746532221145, 0.027854262892022807, 0.02842069481226146, 0.027543675934030343, 0.029150723411650908, 0.029025688300456636, 0.02985310516912229, 0.03139027659493973, 0.031877604569631776, 0.03132156742572969, 0.03185118744651142, 0.03622278715050111, 0.03425343551137762, 0.03370725145991442, 0.032927102038667166, 0.03272079750896573, 0.03484581112451378, 0.03517435002049132, 0.03435893890705266, 0.03438693518858787, 0.03615662323795645, 0.035000219013192356, 0.03304371239225144, 0.03601222530534538, 0.0361470689323056, 0.03545501748925867, 0.034265162357466614, 0.036103308850713486, 0.038759296127366824, 0.0367397700520113, 0.035356600320653665, 0.03543298980429862, 0.03841582683414657, 0.034916958043483194, 0.03970231272241992, 0.035441421056744964, 0.037700950278617605, 0.038147224190664096, 0.03922900611189291, 0.038521386488087936, 0.039103663552930455, 0.03857759339147162, 0.039265084598564974, 0.03719750605089291, 0.0367892907309405, 0.03712436789541034, 0.03811816384294964, 0.03684076538729303, 0.04039373435376836, 0.04279086953924392, 0.037552517820460166, 0.037525665293990554, 0.03965535884960306, 0.040668646395611935, 0.042600769736184714, 0.03986898682301637, 0.04104519735778646, 0.03932767370732385, 0.03780962177613915, 0.03972814991011286, 0.03991857986988189, 0.03455006951165569, 0.035807330652551275, 0.04222714177229543, 0.03920066985660465, 0.04030049287644634, 0.038475673346845735, 0.0403725775885906, 0.04163999825835876, 0.04134298174716659, 0.03916370776443161, 0.039538678442122256, 0.03671933102303778, 0.041964465382051085, 0.04066924564977692, 0.03989584624797953, 0.03614948305737713, 0.040469102546601424, 0.04412786236442403]
mse_error = [0.2036059484495255, 0.08898355668343302, 0.003022849631897472, 0.0023785666144064284, 0.0016638951593090348, 0.0017369637796574976, 0.0010043279083900031, 0.002248244772402276, 0.003719064835695651, 0.005129585239165308, 0.004685401570227337, 0.004194426872851503, 0.0049355481305377435, 0.004716164797376333, 0.004978256328950552, 0.005546676363013585, 0.00493435335626001, 0.006211594383929703, 0.004180109239144672, 0.005211894683719053, 0.003633033734061306, 0.004737419277306016, 0.004272563867119472, 0.004895945175295852, 0.005489584148576301, 0.004812099316442487, 0.00454558439484298, 0.007141732723382658, 0.006591477085829919, 0.006345204873918814, 0.006626972387436458, 0.006687968893738768, 0.00553182504661853, 0.00807696028667403, 0.00797624772545991, 0.007354509325428888, 0.006787593532912394, 0.006419219226430026, 0.007344461969728334, 0.005996295691716949, 0.0075320503006296195, 0.005640209474286529, 0.007876639763852541, 0.009113462996927254, 0.007849235191549732, 0.0047456471706064695, 0.008745485644185145, 0.0064240675390409814, 0.0063708140269894175, 0.0074289081886523625, 0.01063624688987125, 0.006328185278389007, 0.007515012352555318, 0.007541438715732514, 0.008240388007788106, 0.0057346955637127585, 0.009971905492430716, 0.007989155862204876, 0.008323548661086609, 0.009477687552347404, 0.008727858364857856, 0.010411976439838693, 0.010324558056145195, 0.008549750367078416, 0.009262402682897129, 0.007767394597759823, 0.008488833068236992, 0.007653561586825098, 0.0053512225657051295, 0.005731105311246774, 0.010541062283324435, 0.008059732295393892, 0.007894591899159347, 0.007627253116416049, 0.00886314110369288, 0.010068082144225388, 0.008346192726366481, 0.011083436683094344, 0.010451441734366776, 0.01005594999611937, 0.010373945388514614, 0.01000951568609509, 0.008739658636411348, 0.006905234594197227, 0.004781102881296881, 0.01015451175520684, 0.0108946820732459, 0.009205140765085381, 0.00829049256829349, 0.009130159864525754, 0.00832248258793906, 0.010389429897457563, 0.007816674978255169, 0.011665795596528607, 0.008078124858577986, 0.009190643716047511, 0.011040491604475254, 0.009506197382787853, 0.008966445043670177, 0.008748865094141597, 0.012636541090632964]
training_mse = [0.2471470246091485, 0.1357072925195098, 0.10224707126617431, 0.0875707095488906, 0.07121262568980455, 0.09387037195265294, 0.0613532766699791, 0.05355449579656124, 0.06569835413247346, 0.07200298514217138, 0.06173917716369033, 0.04537894111126661, 0.04479813612997532, 0.0436046339571476, 0.05044718123972416, 0.044243433885276316, 0.04105717409402132, 0.045799551904201506, 0.043663221411406994, 0.039153918623924255, 0.03906910018995404, 0.04286310207098722, 0.04064696654677391, 0.04644365618005395, 0.034927446767687796, 0.03620277438312769, 0.04338376550003886, 0.034793841652572155, 0.03412620695307851, 0.03510586246848106, 0.037477971613407136, 0.036432528588920834, 0.03188027748838067, 0.032555905636399984, 0.03322914196178317, 0.03238504156470299, 0.033660374023020266, 0.035248530376702544, 0.0331010109744966, 0.03020956553518772, 0.03197336178272962, 0.03360338164493441, 0.029579117335379122, 0.030522486940026284, 0.03185604121536016, 0.029891329631209374, 0.03099654307588935, 0.03182272305712104, 0.03186836736276746, 0.030987968388944864, 0.02931287856772542, 0.03114809123799205, 0.032361348532140254, 0.03172022271901369, 0.028461593855172395, 0.028291675075888634, 0.029319265112280845, 0.028528594877570868, 0.029016394261270763, 0.028787787538021803, 0.030074402689933777, 0.028602588456124067, 0.029022719524800778, 0.028328416217118502, 0.02561180517077446, 0.02682715216651559, 0.02682448076084256, 0.027983114402741194, 0.0258221042342484, 0.027489471063017846, 0.0290678427554667, 0.02747350661084056, 0.027471069153398274, 0.02730065481737256, 0.025447931420058013, 0.02693046610802412, 0.025259771011769773, 0.027311636973172425, 0.027617321629077197, 0.028231590427458285, 0.02618498168885708, 0.025837603583931924, 0.02520609200000763, 0.025299943704158067, 0.026465628109872342, 0.02582302186638117, 0.025461991783231497, 0.02538913572207093, 0.02600770639255643, 0.02553380550816655, 0.026902954373508692, 0.02468004748225212, 0.02470331350341439, 0.025312031898647547, 0.024457885511219503, 0.02444304237142205, 0.02512458134442568, 0.024646881874650715, 0.024989741388708353, 0.02617455516010523, 0.024091761931777]
training_mse_error = [0.22269689481378419, 0.17484236315243912, 0.08347912889155093, 0.12343162038085212, 0.05578026644410315, 0.08263691042923653, 0.02128271774765804, 0.02979743643949471, 0.03560032805085871, 0.06329529819635381, 0.049234953718414694, 0.011970737530087745, 0.012170823003438669, 0.011121557451638304, 0.01505860193993589, 0.02129313144822305, 0.007145112949509201, 0.01568514430987502, 0.0152496707078582, 0.008070964312327596, 0.00802797011380855, 0.013659603125711548, 0.01687143321323431, 0.02036017623847816, 0.009535148201359528, 0.008374805551206099, 0.014095451371535676, 0.006652282581565735, 0.009237673094998467, 0.0069161548824276265, 0.014676195564803264, 0.01148328506896877, 0.006743734680652225, 0.01000485702197307, 0.005842126010451506, 0.006111962863371269, 0.0066811941974572716, 0.009966892141686967, 0.008803856559745934, 0.003120862975876172, 0.005822455159291889, 0.009916813425175567, 0.003885992081002885, 0.006765881735197785, 0.00932477566105655, 0.005137959118722171, 0.00737502731425173, 0.006479640249754462, 0.009935749326611981, 0.00719455086188247, 0.005718061096193705, 0.006412380473451642, 0.011900014907970421, 0.008855065794271597, 0.004636127408431323, 0.004821733665014622, 0.005890742215159718, 0.0055183576522499994, 0.004238713489893228, 0.004093908598986564, 0.009831904891645512, 0.005738141495901986, 0.006569852568891401, 0.0076247368276551935, 0.0029092599343313523, 0.00746297219521255, 0.002467615132229129, 0.0065662132607650385, 0.002182917827414866, 0.004573804164413717, 0.005749919761068893, 0.005260978456117582, 0.005211898433385721, 0.004629764117626902, 0.0024178518933268794, 0.005507059373982187, 0.003081159950192199, 0.004149571791629831, 0.005628523876651162, 0.004097379819937871, 0.00358597978540356, 0.0025159997983755715, 0.002836108531982021, 0.001906107256994078, 0.003414450965747351, 0.002868115348365252, 0.003399261371337064, 0.0033930377532394944, 0.0035811245233818693, 0.003046340059286404, 0.004827819326147339, 0.003439178740005228, 0.0026182285583638636, 0.003481489500802981, 0.0021692670038429714, 0.0037988819333473105, 0.004040490667083096, 0.002495546192277224, 0.0027375961459952794, 0.004616752680903049, 0.0019540105519813738]


plt.figure()
plt.errorbar(EPOCHS[2:], mse_data[2:],yerr = mse_error[2:],marker='o',ls = '-', capsize = 5, label = 'testing')
plt.errorbar(EPOCHS[2:], training_mse[2:],yerr = training_mse_error[2:],marker='x',ls = '--', capsize = 2, label = 'training')
plt.title('FFN HN: (4, 2), Batch size 300, Learning rate 0.001')
plt.xlabel('Epoch number')
plt.ylabel('Averaged MSE')
plt.ylim(0.02,0.06)
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()

LR = np.round(np.logspace(-4, -2, num = 30),6)
mse_data = [0.03279353808936645, 0.03362419901943416, 0.03182289801504729, 0.0314641209782867, 0.03145174650863929, 0.0316627103468053, 0.031229310465474114, 0.031440765670853035, 0.030649955574057357, 0.030534074650351177, 0.029993427781908822, 0.028133923564898, 0.02862491328613459, 0.025609742551332648, 0.02678211327033162, 0.029352411774323194, 0.029101962956749827, 0.02991172928193562, 0.03080749328482163, 0.03344426313992888, 0.03319981250388244, 0.0326072708622661, 0.03629111148788079, 0.034990187183760645, 0.04239319376799393, 0.033886170466243384, 0.04268059103923612, 0.035683595178724395, 0.03874286302292414, 0.037475071891160464]
mse_error = [0.001185128324349553, 0.006885429648526966, 0.0011012379114290734, 0.0011517574777917417, 0.0014232379834378057, 0.001490093280836886, 0.0006150067713428596, 0.001178973190728371, 0.0026660123933439648, 0.0033648689083462598, 0.003624934722863058, 0.0052005438369203145, 0.004763057212174746, 0.004621128253093569, 0.005634960067402999, 0.004752584238293961, 0.006791208722328859, 0.005991597682749933, 0.00485222874842201, 0.005971687163775376, 0.006513770340728259, 0.005108037841932888, 0.006845891077009942, 0.006758803715716018, 0.009517822244778939, 0.0057069680062222915, 0.013569005569726202, 0.0073029446849354465, 0.005248415988902863, 0.008069337387860996]
training_mse = [0.0795401519164443, 0.08706062380224466, 0.08520014397799969, 0.0828845452517271, 0.07236881945282221, 0.0994593957439065, 0.06491892319172621, 0.055800089798867705, 0.06911526713520288, 0.0747592518106103, 0.06233617067337036, 0.045058912597596645, 0.04372835457324982, 0.04142068410292268, 0.04606110882014036, 0.04004099806770682, 0.03649041438475251, 0.03828351842239499, 0.03566156690940261, 0.03221712876111269, 0.030868414975702762, 0.03171022143214941, 0.029429150559008122, 0.03101239539682865, 0.02565089166164398, 0.02610185295343399, 0.027252095099538565, 0.024237917084246875, 0.02360402662307024, 0.023770920187234878]
training_mse_error = [0.053049158956477174, 0.09397359441008894, 0.05996116134436064, 0.1103859387056385, 0.05691750239469665, 0.08984097351878707, 0.024357372562953603, 0.033419875812756204, 0.03940728795492705, 0.06676813486990207, 0.049647161241402435, 0.01141831484727497, 0.01098315389411378, 0.009608528277128185, 0.012154728080011381, 0.01553176134280283, 0.0057404608237632245, 0.010482688341148443, 0.009776129786623148, 0.00467569072782473, 0.004377097610245415, 0.006927221026501145, 0.0074368074347854335, 0.008110113813729619, 0.004181784701474895, 0.0035445089462807066, 0.0046137861226679035, 0.002187939441926715, 0.0023296963264315003, 0.0027873779414894226]

plt.figure()
plt.errorbar(LR, mse_data,yerr = mse_error,marker='o',ls = '-', capsize = 5, label = 'testing')
plt.errorbar(LR, training_mse,yerr = training_mse_error,marker='x',ls = '--', capsize = 2, label = 'training')
plt.title('FFN HN: (4, 2), Batch size 300, Epochs 200')
plt.xlabel('Learning rate')
plt.ylabel('Averaged MSE')
plt.ylim(0.02,0.09)
plt.xscale('log')
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()


EPOCHS = np.linspace(20,1000,99)
mse_data = [0.04734174838715017, 0.03967290915011079, 0.054474925466428195, 0.03354522656099691, 0.032750951449879724, 0.033639535721187094, 0.03226448012611714, 0.031458055843337246, 0.031154168508313797, 0.03125640162777986, 0.03098466214291721, 0.030823261978830722, 0.03087068012665952, 0.03062484117030236, 0.030470530250257272, 0.03037853959893765, 0.030769148035711163, 0.03042011306821869, 0.030392613274214837, 0.03013622612814722, 0.029831910358943052, 0.030352098142251326, 0.03029012069477483, 0.029767372079772515, 0.029610559324073332, 0.030214994770165565, 0.029287679810063666, 0.02910722661342894, 0.02678817616510225, 0.02701378281322689, 0.027103733893419545, 0.026740718389733, 0.026106529470047362, 0.02590579175561349, 0.024630689758577025, 0.02491460372443768, 0.02576320260786724, 0.024548198257501187, 0.023649483290632996, 0.024034680784213742, 0.023920927089066034, 0.02473366107286522, 0.021966537596178483, 0.021636663643649377, 0.021997762276813212, 0.021659815793330546, 0.020976508342931487, 0.02206797822928592, 0.0210519001151754, 0.02086034317794089, 0.021585474391622526, 0.02201064471984186, 0.021640045302876002, 0.02216624470811922, 0.021014764143564856, 0.02225412000717312, 0.022163119249454376, 0.022148000559275474, 0.023162462167799214, 0.023169801714273297, 0.022930097722279556, 0.022499472634295185, 0.023483175870604184, 0.023311386891402386, 0.024942852532196125, 0.024121346524254808, 0.024595850658142505, 0.02608036485576784, 0.02496607421134723, 0.02621154054078556, 0.025971864405107535, 0.025771442586876737, 0.02699124281082238, 0.02704885974409773, 0.02747994611832097, 0.02638615900561555, 0.030070776631560353, 0.028917846302759027, 0.027824598943572042, 0.028910813646988397, 0.031290141477649504, 0.029148338025053695, 0.029129477352020545, 0.02973940009376458, 0.029451177673039303, 0.03088683042691817, 0.031124412464728152, 0.03357045144563014, 0.03218450718923781, 0.034919598164538365, 0.031271212994193956, 0.032101301887771445, 0.03280644944050851, 0.03478947894617513, 0.03394847419023227, 0.03558589760955706, 0.033792856287012396, 0.03294444509585089, 0.03531492702987452]
mse_error = [0.03284005813333063, 0.011650077923224285, 0.08063635352481222, 0.0008934772350128009, 0.0009020367718075171, 0.004875519095565213, 0.0017077919538810197, 0.0009469232072413519, 0.000836653781639391, 0.000620735443255283, 0.0004098545459802661, 0.0003011846933545413, 0.0004985636378634834, 0.000498108354282327, 0.0005880844797105085, 0.000450616863998788, 0.0007790904076360248, 0.0007113537095192466, 0.000734918549370064, 0.0009589497175981026, 0.0013507895194220735, 0.0011831557132405443, 0.0011972795377509712, 0.0012831278714553914, 0.001667343998549018, 0.0013465247081718605, 0.0021629002293510383, 0.002119281082782153, 0.0034248152911380723, 0.0035405618135536422, 0.0042465876441987625, 0.0038429441789958278, 0.0032088564733481852, 0.004066860304005813, 0.003024102622316798, 0.004480194287759318, 0.0036419234052773326, 0.003981535607917563, 0.00465947873878505, 0.0036877770037460227, 0.005173353949679124, 0.004581047966140635, 0.00326157126402064, 0.0020418391888003726, 0.0043596088841101185, 0.001387021885946627, 0.001753855856822969, 0.0028697974623399236, 0.0024831196948437093, 0.002099771739063051, 0.00168843096953292, 0.0028102209863739244, 0.0020881408988997325, 0.002173411666563334, 0.0019397311854839835, 0.0019402358331654516, 0.0018284260585642715, 0.0017844263106892376, 0.002989638652625947, 0.002111365685823399, 0.002447863198374933, 0.0024828696847891568, 0.0027901532415948384, 0.003103531661970372, 0.004141432163777161, 0.003133503618953684, 0.003120082873743872, 0.0033631991665127115, 0.003538452612341951, 0.0033750831506271536, 0.0030606835733932676, 0.0030003407322580214, 0.004481126256681378, 0.00487766306433997, 0.003531229458054456, 0.003882716054432485, 0.005388530766213278, 0.0034669992749705534, 0.003964753635966373, 0.005359726434455922, 0.005310922260510353, 0.004503366245118174, 0.006039370637373721, 0.005328885219206511, 0.005318366231358306, 0.005686572518390485, 0.006156636385926698, 0.005223063913463314, 0.005691143322522114, 0.007027701462657825, 0.005662019608849827, 0.0052369186175384224, 0.004609971772135395, 0.004917274329580025, 0.006291420597679648, 0.006286288244350178, 0.006535582821379867, 0.005406364561932455, 0.007201126403091957]
training_mse = [0.09332461468875408, 0.09985567424446344, 0.13260079305619002, 0.06530696842819453, 0.06328123640269041, 0.07905634678900242, 0.06601795218884945, 0.05234985463321209, 0.05205637812614441, 0.06126999743282795, 0.053582290187478065, 0.056344916485249995, 0.06389849726110697, 0.05632280688732862, 0.054623213969171046, 0.047376245632767676, 0.05430733319371939, 0.05111165437847376, 0.052886455319821835, 0.0525103434920311, 0.04404078982770443, 0.05091199297457934, 0.045285606756806374, 0.0506611717864871, 0.0479360394179821, 0.04677895717322826, 0.05060812756419182, 0.04612294510006905, 0.047843429073691365, 0.04603526219725609, 0.04064181968569756, 0.04417486544698477, 0.04247566517442465, 0.04685113485902548, 0.0465615289285779, 0.04316373486071825, 0.048110387101769445, 0.04641578681766987, 0.041831599175930025, 0.040656873025000095, 0.04371986426413059, 0.04639798738062382, 0.04067119546234608, 0.04073742441833019, 0.03828531317412853, 0.03913263510912657, 0.039360102452337745, 0.037803889252245426, 0.04200086127966642, 0.03754426091909409, 0.03843855354934931, 0.039634148217737676, 0.037768426537513736, 0.03745096810162067, 0.03858264163136482, 0.03965893741697073, 0.037819499149918556, 0.03773048790171742, 0.03519760863855481, 0.03638464482501149, 0.03524260818958282, 0.035948234423995015, 0.03541606301441789, 0.03767287451773882, 0.03538478445261717, 0.03639208720996976, 0.03639974463731051, 0.03463245555758476, 0.03309177299961448, 0.03511810824275017, 0.0359632876701653, 0.035304094478487966, 0.036809560377150774, 0.03403357015922666, 0.03442646889016032, 0.0341180419549346, 0.03273958964273334, 0.03294397760182619, 0.03445685030892491, 0.03449446782469749, 0.03354066023603082, 0.032463529333472255, 0.0344205348752439, 0.032370210997760294, 0.03402820359915495, 0.03236265191808343, 0.032492695935070516, 0.032029814831912515, 0.03493634220212698, 0.031048497650772333, 0.03643275909125805, 0.031189018674194813, 0.030411931220442057, 0.03197598624974489, 0.030746579449623824, 0.031048831716179847, 0.03117921855300665, 0.03188004214316607, 0.03378493580967188]
training_mse_error = [0.09014494706019517, 0.07975301785289968, 0.20405502350603547, 0.022653214681270784, 0.042190416685387484, 0.07928779545600313, 0.04682309395369973, 0.022121876535176332, 0.017757130702397327, 0.02958064774473637, 0.015241209484981167, 0.026593981516095447, 0.030669470689999534, 0.02051710610515365, 0.021698825524087882, 0.010589135030800813, 0.023824224878038675, 0.03336646425044454, 0.018787851158923365, 0.02948946395615807, 0.004442915511545974, 0.01851796952810034, 0.012643817093062495, 0.014256429377065366, 0.018868715935342208, 0.010449919183792246, 0.01655415581108837, 0.010379362593358506, 0.024141316754206114, 0.012383818464086975, 0.0022986860410037797, 0.008735243492379299, 0.005513557349072855, 0.018374062024170208, 0.009776103646813705, 0.009215378520170268, 0.016892323384889385, 0.02054970089425478, 0.009085570485173226, 0.005849736120650909, 0.00865414723414079, 0.012475030226488229, 0.007888307475613634, 0.006295199972131088, 0.003945325895197247, 0.006802070045355089, 0.00814757731143584, 0.004416094189310168, 0.009541045331725607, 0.003494270000514171, 0.012497094944598016, 0.00689205714153433, 0.006235923832708723, 0.006221166646363362, 0.007248201381099476, 0.006131676020662873, 0.006104364676989693, 0.005791589487571721, 0.0036359908110640466, 0.0037773825316097594, 0.00419613785887814, 0.0037380327900340643, 0.005711043483716014, 0.00986972729825112, 0.005074243328659021, 0.003960091445129921, 0.008191928915071069, 0.0066138736472754955, 0.002332947407646218, 0.006188646842435215, 0.006441056340036744, 0.004914859402406158, 0.013258399681313388, 0.003418687859107163, 0.004833099014718983, 0.0037570845086247148, 0.006311783073370572, 0.004803318147224295, 0.005081544943336356, 0.0069534715492388195, 0.005379482464233284, 0.0029960482198198688, 0.004504661971141427, 0.00386969920363526, 0.007182172253428585, 0.003960348410841794, 0.004397826035605482, 0.0038873406720627826, 0.0072704240797648815, 0.003995724471636083, 0.010706895575970723, 0.0028777170654659583, 0.0022732856959854966, 0.004427322281063476, 0.003416364020658424, 0.007463585722168735, 0.004744702423713264, 0.0037056231115821596, 0.007579915000824929]

plt.figure()
plt.errorbar(EPOCHS, mse_data,yerr = mse_error,marker='o',ls = '-', capsize = 5, label = 'testing')
plt.errorbar(EPOCHS, training_mse,yerr = training_mse_error,marker='x',ls = '--', capsize = 2, label = 'training')
plt.title('FFN HN: (8, 4), Batch size 200, Learning rate 0.00015')
plt.xlabel('Epoch number')
plt.ylabel('Averaged MSE')
plt.ylim(0.015,0.07)
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()


mse_data = [0.026697363318630686, 0.02581630795344589, 0.03554831240844731, 0.04800077822051162, 0.05184483503208322, 0.05154359979047329, 0.06654522457058425, 0.06599733985856415, 0.06996965691896281, 0.05892294641060279, 0.08158118983048787, 0.07212232909594626, 0.09124277293092245, 0.1213897575751702, 0.09348575596128253, 0.09563072618425152, 0.13129847069313597, 0.10735684330639142, 0.09413486818677283, 0.13624679383140442, 0.1274350791607299, 0.11900685132043684, 0.10355289887371064, 0.1065430958390616, 0.12425642172792402, 0.07763353997539492, 0.09096013452229632, 0.15280047087134696]
mse_error = [0.004111181703380451, 0.0034413623260789367, 0.008079445978644155, 0.013997861955886594, 0.014617467806298478, 0.009491654105516627, 0.02016128622568335, 0.01904269385677876, 0.031098956484549148, 0.025575673155038916, 0.024919344866947617, 0.03192267568325498, 0.05005397166283505, 0.07358093053861724, 0.035661575705264875, 0.037937468847589, 0.05558953100477951, 0.07208769774666739, 0.036567224402213945, 0.12354819036349873, 0.0723058601197265, 0.10946298756726744, 0.05612093704626591, 0.07578750288478536, 0.1085233156315217, 0.03032752651716724, 0.039841309968883125, 0.23604486060072014]
training_mse = [0.04126206077635288, 0.034570793807506564, 0.032624262385070324, 0.027545891329646112, 0.025246998760849237, 0.02528687184676528, 0.023454382177442314, 0.022045863792300224, 0.021549700293689966, 0.02196140382438898, 0.018826193641871214, 0.01759049566462636, 0.01689664926379919, 0.016682940116152168, 0.015941264946013688, 0.015552551159635187, 0.015585189871490001, 0.015023563243448734, 0.01483064740896225, 0.013886572048068046, 0.013415176328271628, 0.013215888943523168, 0.012874068692326546, 0.013633384043350816, 0.01383915487676859, 0.013605175167322158, 0.013995013572275639, 0.014360577752813696]
training_mse_error = [0.006621748534992276, 0.004712301036835695, 0.011127832947300405, 0.0021558148230152495, 0.0022976255799515726, 0.003463456449692331, 0.0025275203230591477, 0.0014806626164276953, 0.0014945583577330452, 0.0017445457055200377, 0.0007716843281981166, 0.00066844850406522, 0.000827189237248453, 0.0007027902535023178, 0.0006338280936856876, 0.0005802411638713461, 0.0006511268896528874, 0.0004592927539109914, 0.0006176448410068086, 0.0006690748447080679, 0.0005757688190329606, 0.0007450470908428517, 0.0005711914301690072, 0.0013606619353524984, 0.0007661965009317489, 0.0007835327793100348, 0.0014624233497174373, 0.0013645492825019984]
LR = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]

plt.figure()
plt.errorbar(LR, mse_data,yerr = mse_error,marker='o',ls = '-', capsize = 5, label = 'testing')
plt.errorbar(LR, training_mse,yerr = training_mse_error,marker='x',ls = '--', capsize = 2, label = 'training')
plt.title('FFN HN: (8, 4), Batch size 200, Epochs 510')
plt.xlabel('Learning rate')
plt.ylabel('Averaged MSE')
plt.ylim(0.0,0.175)
plt.xscale('log')
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()

LR = np.round(np.logspace(-4, -3, num = 30),6)
mse_data = [0.02650588475107186, 0.023671787889784746, 0.022642877124818283, 0.023113624945119267, 0.02152676262426973, 0.021712748421473322, 0.023457893008792773, 0.025298589814066453, 0.026094503569698096, 0.028664955573855445, 0.030410159845099073, 0.031896701125056916, 0.03689269319950954, 0.03565259902988712, 0.03509806496912292, 0.04166968396372752, 0.04184039758350995, 0.04633680726059724, 0.04681845527467908, 0.051813555412380495, 0.05269514791921159, 0.05366784866498914, 0.06357607718713768, 0.057399221600700864, 0.059930108072501286, 0.06260562318329595, 0.05590998521560573, 0.06502386767034578, 0.07853361409972473, 0.0829535825637144]
mse_error = [0.0043264404815558195, 0.003797579713067736, 0.0037941983635670366, 0.0032596700293023627, 0.0017233043012318669, 0.0021936981635121916, 0.0023433659591064393, 0.002467581489417919, 0.0038038758038368256, 0.006085421407227052, 0.0048266710760804495, 0.005483323613208714, 0.0046045076776928835, 0.007424858561805887, 0.006187205054617042, 0.008452348527178025, 0.005024638745339472, 0.008410016314557182, 0.009943430637127972, 0.016859159070108317, 0.014007025933991729, 0.020117205377545574, 0.02349881419244376, 0.026451140887606736, 0.015082960982491723, 0.013208030820661508, 0.019440371765471396, 0.014997607056877254, 0.03876609319118566, 0.04251736583182196]
training_mse = [0.04093533158302307, 0.041759468242526056, 0.0461768489331007, 0.038774438016116616, 0.03743557427078485, 0.03999760616570711, 0.03717963779345155, 0.03388704126700759, 0.03298648931086064, 0.034565795119851826, 0.03239234937354922, 0.03256552945822477, 0.03287505293264985, 0.03124660961329937, 0.03051395546644926, 0.0280302032828331, 0.028841822408139706, 0.027741020638495685, 0.027525639813393354, 0.026458257902413608, 0.024655394442379474, 0.025765038561075926, 0.024060557875782252, 0.024647681415081023, 0.023375967983156443, 0.023368211183696985, 0.023294714279472828, 0.022192446421831845, 0.021770512871444225, 0.021339308936148882]
training_mse_error = [0.00666989325903208, 0.007336948046025568, 0.024637491206490225, 0.0038741530808540543, 0.0060943162289199505, 0.011917557014428577, 0.0077288447678130244, 0.00401696440529596, 0.0039021229993388834, 0.0056473905377016275, 0.003133854787352074, 0.005342263549356142, 0.006555869004669477, 0.004340641877585667, 0.004452537064764439, 0.002410105660929257, 0.005109479980583639, 0.005881083969120903, 0.0037615376616452513, 0.005413169592033886, 0.0013584270582773165, 0.004203974042414541, 0.0025098501495550807, 0.0027361865328359514, 0.003090492645751292, 0.0021387554890797186, 0.002906721529494617, 0.0017668471219792466, 0.0032795156424654656, 0.0020790211848265637]

plt.figure()
plt.errorbar(LR, mse_data,yerr = mse_error,marker='o',ls = '-', capsize = 5, label = 'testing')
plt.errorbar(LR, training_mse,yerr = training_mse_error,marker='x',ls = '--', capsize = 2, label = 'training')
plt.title('FFN HN: (8, 4), Batch size 200, Epochs 510')
plt.xlabel('Learning rate')
plt.ylabel('Averaged MSE')
plt.ylim(0.0,0.1)
plt.xscale('log')
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()

BS = np.linspace(50,800,16)
mse_data = [0.0407452775569904, 0.02963905354777397, 0.023548682938952424, 0.02157234645895914, 0.02381739229306965, 0.024108597172928303, 0.02443379206852133, 0.029188990206419855, 0.027204975245631348, 0.02899634219608664, 0.02899275507325487, 0.029911167513381395, 0.03063860802028207, 0.03050095505960946, 0.030525213009251435, 0.03079287768082173]
mse_error = [0.0072770815872756625, 0.006042866005088317, 0.0028802160090875746, 0.0017897926260493763, 0.003430515437347466, 0.004049753793510791, 0.003232532164260383, 0.0027535716149659365, 0.0037823021043768804, 0.002291179117843661, 0.0024042327925148546, 0.0014059420840074745, 0.0006322859633749977, 0.0012898163953053944, 0.0013192425268102254, 0.0006090960770368539]
training_mse = [0.02739242126699537, 0.031986206537112594, 0.035048862686380744, 0.04409482912160456, 0.03894150326959789, 0.0384934744797647, 0.04863976244814694, 0.0474891874473542, 0.045185150345787406, 0.04402091074734926, 0.04463939811103046, 0.0523488272447139, 0.04883190616965294, 0.05155291361734271, 0.05523005686700344, 0.06229353626258671]
training_mse_error = [0.0017708227494946619, 0.0038114843016643727, 0.003597782698819753, 0.02194807978623369, 0.004039059507743491, 0.005785178529511355, 0.021982949063638634, 0.011321578542643146, 0.01563399988977092, 0.011010800600482705, 0.010144027676850367, 0.021580147889602636, 0.007626909389924463, 0.014659368859829288, 0.02683786548440445, 0.029484213370561434]

plt.figure()
plt.errorbar(BS, mse_data,yerr = mse_error,marker='o',ls = '-', capsize = 5, label = 'testing')
plt.errorbar(BS, training_mse,yerr = training_mse_error,marker='x',ls = '--', capsize = 2, label = 'training')
#plt.title('FFN HN: (8, 4), Learning rate 0.00015, Epochs 510')
plt.xlabel('Batch size')
plt.ylabel('Averaged MSE')
#plt.ylim(0.0,0.1)
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()


