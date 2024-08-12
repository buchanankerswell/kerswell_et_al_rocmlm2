#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.odr import ODR, Model, RealData,Data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set sys paths to call HyMaTZ submodules !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
hymatz_dir = os.path.join(current_dir, 'HyMaTZ')
sys.path.insert(0, hymatz_dir)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HyMaTZ !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from HyMaTZ.GUI_cal_MP import Phase_diagram
from HyMaTZ.Mineral_Physics.Velocity_calculator import Velocity_calculator
from HyMaTZ.Mineral_Physics.Stix2011data import (ab, an, sp, hc, fo, fa, mgwa, fewa, mgri,
                                                 feri, en, fs, mgts, odi, di, he, cen, cats,
                                                 jd, hpcen, hpcfs, mgpv, fepv, alpv, capv,
                                                 mgil, feil, co, py, al, gr, mgmj, jdmj, qtz,
                                                 coes, st, mppv, fppv, appv, pe, wu, mgcf,
                                                 fecf, nacf, ky, neph, OL_, WA_, RI_)
from HyMaTZ.Mineral_Physics.Solidsolution import (c2c, CF, Cpx, Gt, Aki, Wus, O, Opx, Pl, Ppv,
                                                  ppv, Pv, Ring, Sp, Wad, OLwater, WAwater,
                                                  RIwater)

#######################################################
## .1.          HyMaTZ Regression class          !!! ##
#######################################################
class Regression(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, name=None):
        """
        """
        self.name = name
        self.key, self.content = self._read_data()
        if self.name == "olivine":
            self.pressure_deri = [4.217, 1.462]
        if self.name == "wadsleyite":
            self.pressure_deri = [4.322, 1.444]
        if self.name == "ringwoodite":
            self.pressure_deri = [4.22, 1.354]
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # regression plane !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _regression_plane(self, x, x_error, y, y_error, z, z_error):
        """
        """
        X = []
        Y = []
        Z = []
        X_error = []
        Y_error = []
        Z_error = []
        for i in range(len(z)):
            try:
                Z.append(float(z[i]))
                X.append(float(x[i]))
                Y.append(float(y[i]))
                X_error.append(float(x_error[i]))
                Y_error.append(float(y_error[i]))
                Z_error.append(float(z_error[i]))
            except:
                pass
        if sum(X_error) == 0 or sum(Y_error) == 0 or sum(Z_error) == 0:
            return None
        data = RealData([X, Y], Z, sx=[X_error, Y_error], sy=Z_error)
        def func(beta,data):
            x,y = data
            a,b,c = beta
            return a*x+b*y+c
        model = Model(func)
        odr = ODR(data, model, [100, 100, 100])
        res = odr.run()
        return res.beta, res.sd_beta

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # read data !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _read_data(self):
        """
        """
        def Stinglist_float(string_list):
            number_list = []
            def average(array):
                num = 0
                sum1 = 1e-3
                for i in array:
                    try:
                        sum1+=float(i)
                        num+=1
                    except:
                        pass
                if num == 0:
                    print(array)
                return sum1 / num
            def string_float(string):
                return_string = ""
                for i in string:
                    if i != " ":
                        return_string +=i
                a=[(x) for x in re.split(r"[()]", string)]
                list1=[]
                for i in a:
                    try:
                        list1.append(float(i))
                    except:
                        pass
                if len(list1) == 1:
                    list1.append("NA")
                if len(list1) != 2:
                    list1=["NA", "NA"]
                return list1
            for i in string_list:
                list1 = string_float(i)
                number_list.append(list1)
            aa=np.array(number_list)
            ave=average(aa[:, 1])
            for i in range(len(aa[:, 1])):
                if aa[:, 1][i] == "NA":
                    aa[:, 1][i] = ave
            return aa[:,0],aa[:,1]
        try:
            address = os.path.join(os.path.dirname(__file__), "HyMaTZ", "Mineral_Physics",
                                   "EXPDATA", self.name + ".txt")
            self.address = address
            file = open(address, "r+")
        except:
            raise Exception(f"No file found at {self.address} !")
        for i in file:
            name=i.split(",")
            break
        dictionary=dict()
        for i in name:
            dictionary[i]=[]
        for line in file:
           line=line.split(",")
           for i in range(len(line)):
                   dictionary[name[i]].append((line[i]))
        file.close()
        a, b = Stinglist_float(dictionary["H2O"])
        self.water_content = a
        self.water_content_error = b
        a, b = Stinglist_float(dictionary["Iron"])
        self.iron_content = a
        self.iron_content_error = b
        a, b = Stinglist_float(dictionary["K"])
        self.K = a
        self.K_error = b
        a, b = Stinglist_float(dictionary["G"])
        self.G = a
        self.G_error = b
        a, b = Stinglist_float(dictionary["Rho"])
        self.Rho = a
        self.Rho_error = b
        return name, dictionary

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # function K !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def function_K(self):
        """
        """
        return self._regression_plane(self.water_content,self.water_content_error,
                                      self.iron_content, self.iron_content_error, self.K,
                                      self.K_error)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # function G !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def function_G(self):
        """
        """
        return self._regression_plane(self.water_content,self.water_content_error,
                                      self.iron_content, self.iron_content_error, self.G,
                                      self.G_error)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # function Rho !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def function_Rho(self):
        """
        """
        return self._regression_plane(self.water_content,self.water_content_error,
                                      self.iron_content, self.iron_content_error, self.Rho,
                                      self.Rho_error)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # function Return !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def Return(self):
        """
        """
        return self.function_K(), self.function_G(), self.function_Rho()

#######################################################
## .2.         HyMaTZ WaterProfile class         !!! ##
#######################################################
class WaterProfile():
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, Depth=None, Pressure=None, Temperature=None, usercontrol=None):
        """
        """
        self.usercontrol=usercontrol
        self.Depth = np.array(Depth)
        self.Pressure = np.array(Pressure)
        self.OL_water = np.ones(len(Depth))
        self.WA_water = np.zeros(len(Depth))
        self.RI_water = np.zeros(len(Depth))
        self.Temperature = np.array(Temperature)
        self.OL=self.usercontrol["OLcoefficient"]
        self.WA=self.usercontrol["WAcoefficient"]
        self.RI=self.usercontrol["RIcoefficient"]
        self.OL_function = self.usercontrol["OL_function"]
        self.WA_function = self.usercontrol["WA_function"]
        self.RI_function = self.usercontrol["RI_function"]
        self.editor_OL()
        self.editor_WA()
        self.editor_RI()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # editor OL !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def editor_OL(self):
        """
        """
        D = self.Depth
        P = self.Pressure
        T = self.Temperature
        self.OL_water = (eval(self.OL_function) + np.zeros(len(D))) * self.OL / 100
        for i in range(len(D)):
            if self.OL_water[i] >= 1.1:
                self.OL_water[i] = 1.1
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # editor WA !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def editor_WA(self):
        """
        """
        D = self.Depth
        P = self.Pressure
        T = self.Temperature
        self.WA_water = (eval(self.WA_function) + np.zeros(len(D))) * self.WA / 100
        for i in range(len(D)):
            if self.WA_water[i] >= 3.3:
                self.WA_water[i] = 3.3
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # editor RI !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def editor_RI(self):
        """
        """
        D = self.Depth
        P = self.Pressure
        T = self.Temperature
        self.RI_water = (eval(self.RI_function) + np.zeros(len(D))) * self.RI / 100
        for i in range(len(D)):
            if self.RI_water[i] >= 2.7:
                self.RI_water[i] = 2.7
        return None

#######################################################
## .3.               HyMaTZ class                !!! ##
#######################################################
class HyMaTZ():
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, mantle_potential=1573, sid="Pyrolite", water_capacity=50, res=300,
                 verbose=1):
        """
        """
        self.res = res
        self.sid = sid
        self.results = {}
        self.verbose = verbose
        self.K = np.zeros(res)
        self.G = np.zeros(res)
        self.Vp = np.zeros(res)
        self.Vs = np.zeros(res)
        self.fo = np.zeros(res)
        self.sys = np.zeros(res)
        self.Rho = np.zeros(res)
        self.mgwa = np.zeros(res)
        self.mgri = np.zeros(res)
        self.Phase_diagram = None
        self.fig_dir = "figs/hymatz"
        self.data_dir = "assets/data"
        self.h2o_profile = np.zeros(res)
        self.water_capacity = water_capacity
        self.mantle_potential = mantle_potential
        self.profile_prefix = f"hymatz-{mantle_potential}K-{self.sid}-{water_capacity}H2O"
        self.profile_out_path = f"{self.data_dir}/{self.profile_prefix}"
        self.waterusercontorl = {"OL_function": "4.0*1e-6*D*D",
                                 "WA_function": "3.3",
                                 "RI_function": "1.7",
                                 "OLcoefficient": water_capacity,
                                 "WAcoefficient": water_capacity,
                                 "RIcoefficient": water_capacity}
        if sid == "Pyrolite":
            self.comp = np.array([3.14, 2.23, 0.33, 49.94, 5.48, 38.85])
        else:
            raise Exception("Unrecognized sid !")
        self.existing_profile = False
        self._check_existing_model()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check existing model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_existing_model(self):
        """
        """
        # Get self attributes
        if os.path.exists(self.profile_out_path):
            try:
                self.existing_profile = True
                df = pd.read_csv(self.profile_out_path, sep="\t")
                self.results = {col: df[col].to_numpy() for col in df.columns}
            except Exception as e:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(f"!!! ERROR in _check_existing_model() !!!")
                print(f"{e}")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                traceback.print_exc()
                return None
        else:
            self.build_profile()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_model(self):
        """
        """
        # Get self attributes
        res = self.res
        sid = self.sid
        comp = self.comp
        mantle_potential = self.mantle_potential
        try:
            self.sys = Phase_diagram(num=res, Model=sid, Model_composition=comp)
            self.sys.Phase_diagram_first_pinciple(mantle_potential)
            for number, phase in enumerate(self.sys.Phase_diagram):
                self.fo[number] = phase[24] / (phase[24] + phase[25] + 1e-11)
                self.mgwa[number] = phase[26] / (phase[26] + phase[27] + 1e-11)
                self.mgri[number] = phase[28] / (phase[28] + phase[29] + 1e-11)
            self.Phase_diagram = np.array(self.sys.Phase_diagram)
            self.h2o_profile = WaterProfile(Depth=self.sys.Depth,
                                            Pressure=self.sys.Pressure,
                                            Temperature=self.sys.Temperature,
                                            usercontrol=self.waterusercontorl)
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _configure_model() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # compute elastic properties !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_elastic_properties(self):
        """
        """
        # Get self attributes
        Pressure = self.sys.Pressure
        h2o_profile = self.h2o_profile
        Temperature = self.sys.Temperature
        Phase_diagram = self.Phase_diagram
        try:
            V_OL = []
            V_WA = []
            V_RI = []
            ol = Regression("olivine")
            wad = Regression("wadsleyite")
            ring = Regression("ringwoodite")
            OL = OL_(ol)
            WA = WA_(wad)
            RI = RI_(ring)
            mineral_list = [an, ab, sp, hc, en, fs, mgts, odi, hpcen, hpcfs, di, he, cen,
                            cats, jd, py, al, gr, mgmj, jdmj, capv, fo, fa, mgwa, fewa, mgri,
                            feri, mgil, feil, co, mgpv, fepv, alpv, mppv, fppv, appv, mgcf,
                            fecf, nacf, pe, wu, qtz, coes, st, an, ky, neph, OL, WA, RI]
            for i in mineral_list:
                i.Clear_Vp_Vs()
            for number, phase in enumerate(Phase_diagram):
                K = []
                G = []
                V = []
                Rho = []
                phase[53] = 0
                phase[54] = 0
                V_OL.append(0.)
                V_WA.append(0.)
                V_RI.append(0.)
                OL.Set_Water_Iron_Condition(
                    h2o_profile.OL_water[number], 1 - self.fo[number])
                WA.Set_Water_Iron_Condition(
                    h2o_profile.WA_water[number], 1 - self.mgwa[number])
                RI.Set_Water_Iron_Condition(
                    h2o_profile.RI_water[number], 1 - self.mgri[number])
                for num, i in enumerate(mineral_list, start=3):
                    if phase[num] != 0:
                        if num==24 or num==25 or num==26 or num==27 or num==28 or num==29:
                            pass
                        else:
                            r, k, g, v, rho = i.EOS(Pressure[number]*1e5, Temperature[number])
                            K.append(k)
                            G.append(g)
                            Rho.append(rho)
                            V.append(phase[num])
                            if i.name == "Olivine":
                                V_OL[number] = phase[num]
                            if i.name == "Wadsleyite":
                                V_WA[number] = phase[num]
                            if i.name == "Ringwoodite":
                                V_RI[number] = phase[num]
                            a, b, c = i.Vp_Vs(Pressure[number] * 1e5, Temperature[number])
                            c *= 1000
                            i.Store_Vp_Vs(a, b, c, phase[1], phase[num])
                KGRho = Velocity_calculator(K, G, Rho, V)
                self.K[number], self.G[number], self.Rho[number] = KGRho.Voigt_Reuss_Hill()
                self.Vp[number] = (np.sqrt((self.K[number] + 4. * self.G[number] / 3.) /
                                           self.Rho[number]) / 1000.)
                self.Vs[number] = np.sqrt(self.G[number]/self.Rho[number]) / 1000.
            V_OL = np.array(V_OL)
            V_WA = np.array(V_WA)
            V_RI = np.array(V_RI)
            OL_water = self.h2o_profile.OL_water
            WA_water = self.h2o_profile.WA_water
            RI_water = self.h2o_profile.RI_water
            h2o = (V_OL * OL_water) + (V_WA * WA_water) + (V_RI * RI_water)
            self.results = {"T": np.round(self.sys.Temperature, 3),
                            "P": np.round(self.sys.Pressure / 1e4, 3),
                            "rho": np.round(self.Rho / 1e3, 3),
                            "Vp": self.Vp.round(3),
                            "Vs": self.Vs.round(3),
                            "h2o": h2o.round(3),
                            "OL_mode": V_OL.round(3),
                            "WA_mode": V_WA.round(3),
                            "RI_mode": V_RI.round(3),
                            "OL_water": OL_water.round(3),
                            "WA_water": WA_water.round(3),
                            "RI_water": RI_water.round(3)}
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _compute_elastic_properties() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # save results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _save_results(self):
        """
        """
        # Get self attributes
        results = self.results
        profile_out_path = self.profile_out_path
        try:
            df_to_save = pd.DataFrame(self.results)
            df_to_save.to_csv(profile_out_path, sep="\t", index=False)
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in _save_results() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # build profile !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def build_profile(self):
        """
        """
        # Get self attributes
        profile_prefix = self.profile_prefix
        try:
            print(f"  Building HyMaTZ profile {profile_prefix} ...")
            self._configure_model()
            self._compute_elastic_properties()
            self._save_results()
        except Exception as e:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"!!! ERROR in build_profile() !!!")
            print(f"{e}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            traceback.print_exc()
        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # plot profile !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_profile(self, target="rho", figwidth=6.3, figheight=5.8, fontsize=28):
        """
        """
        # Get self attributes
        sid = self.sid
        fig_dir = self.fig_dir
        P_hymatz = self.sys.Pressure / 1e4
        P_min = np.nanmin(P_hymatz)
        P_max = np.nanmax(P_hymatz)
        os.makedirs(fig_dir, exist_ok=True)

        if target == "rho":
            target_hymatz = self.Rho / 1e3
            target_label = "Density"
        elif target == "Vp":
            target_hymatz = self.Vp
            target_label = target
        elif target == "Vs":
            target_hymatz = self.Vs
            target_label = target
        else:
            raise Exception("  Unrecognized target !")

        # Set plot style and settings
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["font.size"] = fontsize
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["figure.autolayout"] = "True"

        # Colormap
        colormap = plt.colormaps["tab10"]

        # Plotting
        fig, ax1 = plt.subplots(figsize=(figwidth, figheight))
        ax1.plot(target_hymatz, P_hymatz, "-", linewidth=3, color=colormap(1),
                 label=f"1573 K")
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # Convert the primary y-axis data (pressure) to depth
        depth_conversion = lambda P: P * 30
        depth_values = depth_conversion(np.linspace(P_min, P_max, len(P_hymatz)))

        # Create the secondary y-axis and plot depth on it
        ax2 = ax1.secondary_yaxis(
            "right", functions=(depth_conversion, depth_conversion))
        ax2.set_yticks([410, 670])
        ax2.set_ylabel("Depth (km)")

        plt.legend(loc="lower right", columnspacing=0, handletextpad=0.2,
                   fontsize=fontsize * 0.833)

        plt.title("Depth Profile")

        # Save the plot to a file
        filename = f"{sid}-{target}-hymatz.png"
        plt.savefig(f"{fig_dir}/{filename}")

        # Close device
        plt.close()
        print(f"  Figure saved to: {fig_dir}/{filename} ...")

        return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# main !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    """
    """
    try:
        for w in np.linspace(0, 100, 11, dtype=int):
            model = HyMaTZ(water_capacity=w)
    except Exception as e:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"!!! ERROR in main() !!!")
        print(f"{e}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        traceback.print_exc()
    return None

if __name__ == "__main__":
    main()
