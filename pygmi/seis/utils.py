# -----------------------------------------------------------------------------
# Name:        utils.py (part of PyGMI)
#
# Author:      Patrick Cole
# E-Mail:      pcole@geoscience.org.za
#
# Copyright:   (c) 2020 Council for Geoscience
# Licence:     GPL-3.0
#
# This file is part of PyGMI
#
# PyGMI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyGMI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
"""
Module for miscellaneous utilities relating to earthquake seismology.
"""

import difflib
import os
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pygmi.menu_default as menu_default


class CorrectDescriptions(QtWidgets.QDialog):
    """
    Correct seisan descriptions.

    This compares the descriptions found in seisan type 3 lines to a custom
    list.

    Attributes
    ----------
    parent : parent
        reference to the parent routine
    indata : dictionary
        dictionary of input datasets
    outdata : dictionary
        dictionary of output datasets
    """

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)

        self.indata = {}
        self.outdata = {}
        self.parent = parent
        self.pbar = parent.pbar

        idir = os.path.dirname(os.path.realpath(__file__))
        tfile = os.path.join(idir, r'descriptions.txt')

        self.textfile = QtWidgets.QLineEdit(tfile)

        self.setupui()

    def setupui(self):
        """
        Set up UI.

        Returns
        -------
        None.

        """
        gridlayout_main = QtWidgets.QGridLayout(self)
        buttonbox = QtWidgets.QDialogButtonBox()
#        helpdocs = menu_default.HelpButton('pygmi.grav.iodefs.importpointdata')
        pb_textfile = QtWidgets.QPushButton('Load Description List')


        buttonbox.setOrientation(QtCore.Qt.Horizontal)
        buttonbox.setCenterButtons(True)
        buttonbox.setStandardButtons(buttonbox.Cancel | buttonbox.Ok)

        self.setWindowTitle(r'Correct Descriptions')

        gridlayout_main.addWidget(self.textfile, 0, 0, 1, 1)
        gridlayout_main.addWidget(pb_textfile, 0, 1, 1, 1)

#        gridlayout_main.addWidget(helpdocs, 5, 0, 1, 1)
        gridlayout_main.addWidget(buttonbox, 5, 1, 1, 3)

        buttonbox.accepted.connect(self.accept)
        buttonbox.rejected.connect(self.reject)
        pb_textfile.pressed.connect(self.get_textfile)

    def get_textfile(self, filename=''):
        """
        Get description list filename.

        Parameters
        ----------
        filename : str, optional
            Filename submitted for testing. The default is ''.

        Returns
        -------
        None.

        """
        ext = ('Description list (*.txt)')

        if filename == '':
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self.parent, 'Open File', '.', ext)
            if filename == '':
                return

        self.textfile.setText(filename)

    def settings(self):
        """
        Entry point into item.

        Returns
        -------
        tmp : bool
            True if successful, False otherwise.

        """
        if 'Seis' not in self.indata:
            return False

        tmp = self.exec_()

        if tmp == 1:
            self.acceptall()
            tmp = True

        return tmp

    def acceptall(self):
        """
        Accept option.

        Updates self.outdata, which is used as input to other modules.

        Returns
        -------
        None.

        """

        filename = self.textfile.text()
        with open(filename) as fno:
            tmp = fno.read()

        masterlist = tmp.split('\n')

        data = self.indata['Seis']

        nomatch = []
        correction = []

        for i in data:
            if '3' not in i:
                continue
            text = i['3'].region

            cmatch = difflib.get_close_matches(text, masterlist, 1, cutoff=0.7)
            if cmatch:
                cmatch = cmatch[0]
            else:
#                self.parent.showprocesslog('No match found for '+text)
                nomatch.append(text)
                continue

            if cmatch != text:
#                self.parent.showprocesslog('Correcting '+text+' to '+cmatch)
                correction.append(text+' to '+cmatch)
                i['3'].region = cmatch


        breakpoint()
        self.outdata['Seis'] = data



def main():
    """
    Main

    Returns
    -------
    None.

    """
    mines = ['Far West Rand gold mines',
             'West Rand gold mines',
             'East Rand gold mines',
             'Central Rand gold mines',
             'Free State gold mines',
             'Klerksdorp gold mines',
             'Bushveld platinum mines',
             'Witbank coal field',
             'Highveld coal field',
             'Ermelo coal field',
             'Vryheid coal field',
             'Klip River coal field',
             'Sishen manganese field',
             'Kalahari manganese field',
             'Aggeneys mine',
             'Lime Acres mine',
             'Phalaborwa mine',
             'Ga-Mapela mine',
             'Letseng mine',
             'Cullinan mine',
             'Venetia mine',
             'Piketberg quarry',
             'Riebeeck West quarry',
             'Worcester quarry',
             'De Hoek quarry',
             'Fisantkraal quarry',
             'Sasolburg coal mines']

    tmp = 'Far Wes1 Ranf gld mnes'

#    aaa = difflib.get_close_matches(tmp, mines, 1)


    nomatch = ['Kurumanheuwels area', 'Bethlehem area', 'Kguer Nat. Park', 'Ellisras area', 'Babanango area', 'Vaalkop mine', 'Danhauser area', 'Riebeek-Kasteel area', 'Sada, Eastern Cape', 'Wepener area', 'Hopefield area', 'Madadeni mine', 'Border of Botswana', 'Giyani area', 'Mziwabantu area', 'Galeshewe area', 'Kokosi area', 'Ryedale manganese mine', 'Marguard area', 'Koster area', 'Germiston area', 'Marydale area', 'Knersvlakte', 'Balfour area', 'Springbok area', 'Ixopo area', 'Bela-Bela area', 'Springbok Flats coal field', 'Koringberg area', 'Zeerust mines', 'Ga-Mogopa area', 'Witrivier area', 'Groot Marico area', 'Lydenburg area', 'Waterford area', 'Marchand area', 'KwaBhubesi area', 'Phelindaba area', 'Silvermyn area', 'Bababnango area', 'Marquard area', 'Sishen area', 'Aldam area', 'Soutpansberg', 'Onverwacht area', 'Nhlazatje area', 'Bela Bela area', 'Benede area', 'GaMathabatha area', 'Great Karoo', 'Basterspad area', 'Koppies Area', 'Campbell area', 'Winburg area', 'Kendal area', 'Mmabatho area', 'Nietverdiend mines', 'Brandford area', 'Esperanza area', 'Indian Ocean Triple Junction', 'Dewitsdorp area', 'Scottburgh area', 'Paulpietersburg area', 'Klein-Karoo', 'Groenfontein mine', 'Offshore West coast', 'Bodibe area', 'Lyden area', 'Lekgalameetse area', 'Goegap area', 'Sani Pass area', 'Villiersdorp area', 'Mtubatuba area', 'Olifants area, Kruger National Park', 'Tankwa Karoo', 'eZamokuhle area', 'Kanoneiland area', 'Machadodorp area', 'Douglas area', 'Kwadelamuzi area', 'Parys area', ':  Ceres area', 'GaMaja area', 'Off-coast, Mozambique', 'Moorreesburg area', 'Central Botswana', 'Middleton area', 'Middel Koegas area', 'Comores', 'Bela - Bela area', 'Paternoster area', 'Pilansberg Nat. Reserve', 'Lake Albert Region, Congo', 'Soshanguve area', 'Witkop colliery', 'Barberton area', 'Sterkfontein Dam', 'Cassel area', 'Hutchinson area', 'Tlholong area', 'Eastern Botswana', 'Luxwesa area', 'KwaMandlenkosi area', 'Cathcart area', 'Dibete - Botswana', 'Seshego area', 'Reitz area', 'George Area', 'Ongegund mine', 'Jan Kempdorp area', 'Sipola area', 'Kamieskroon area', ': East of Sishen Manganese area', 'Wegdraai area', 'Oudshoorn', 'Marapong mine', 'kruger Nat. Park', 'Vryheid area', 'Terra Firma area', 'Groot-Marico area', 'Vhembe area', 'TableBay area', 'KwaSithole area', 'Leydsdorp area', 'Dumisa area', 'Tafelkop area', 'Magaliesberg area', 'Magaliesburg area', 'Gordonia', 'Waterberge', 'Kalkfontein dam', 'Aggenyes area', 'Barrydale area', 'Atherstone area', 'Loskop Dam area', 'Kagga-kamma area', 'Reddersburg Area', 'Heidelberg area', 'Elandsbaai area', 'Loskopdam area', 'Siyathemba area', 'KuNdayingana area', 'Zithobeni area', 'Kwa-Sithole area', 'Pilansberg Nat. Res.', 'Winterton area', 'Hennenman area', 'Sehlakwane area', 'Herbertsdale area', 'Beenbreek area', 'Kwa-nonzane area', 'Thabazimbi mine', 'Kimberley area', 'Sishen mine', 'Tafelkop  area', 'Putsonderwater area', 'eKuyukeni area', 'Kestell area', 'Three Sisters area', 'Orapa mine', 'Paulpietrsburg area', 'Dannhauser area', 'Vanuatu', 'GaMolekwa area', 'Bellsbank mines', 'Vetpan mine', 'Krone mine', 'Weppener area', 'Kimberley  area', 'Granaatboskolk Area', 'Serake area', 'Marapjane area', 'Jonkersberg area', 'Witleigat area', 'Greysdorp area', 'Machadadorp area', 'Stoffberg area', 'Delmas area', 'Loskop dam area', 'Westonaria area', 'GaPhamadi area', 'Springs coal mine', 'Voorspoed quarry', 'Lichtenburg area', 'Kalkfontein Dam', 'Sterkfontein Dam area', 'Middelburg area', 'Central Zimbabwe', 'Vrede area', 'Tankwa-karoo area', 'Senotlelo area', 'Ntsikeni National Park', 'Ceres Area', 'Siyabuswa area', 'Madinoga mine', 'Pilansberg Nat Res.', 'Groor-Marico area', 'Granaatboskolk area', 'Estcourt area', 'Belford area', 'Hertz quarry', 'KwaDelamuzi area', 'Qeduni area', 'Grahamstown area', 'Wolseley area', 'Dewetsdorp area', 'KwaVikinduku area', 'Orania area', 'Maroelakop mine', 'Zeerust area', 'Paarl area', 'Offshore Mozambique', 'Constantia mine', 'Kaiingveld', 'Sekuruwe area', 'Zastron area']
    correction = ['Niekershoop area to Niekerkshoop area, N. Cape', 'Wellington area to Wellington area, W. Cape', 'Groblersdal mine to Groblersdal area', 'Baberton area to Dendron area', 'Herold area to Borolo area', 'Evander area to Pofadder area', 'Ga-Mapela area to Ga-Mapela mine', 'Kgautswane area to Ratswini area', 'Koffiefontein to Koffiefontein area', 'Mozambique area to Mozambique', 'Calvinia area to Calvinia area, N. Cape', 'Umgayi area to Zigagayi area', 'Mogohlwaneng to Mogohlwaneng area', 'Middelpos area to Middelpos area, N. Cape', 'Verneukpan area to Verneukpan area, N. Cape', 'St.Helena Bay to Off coast St. Helena Bay', 'Sebayeng area to Sebayeng area, Limpopo', 'Far West Rand gold mies to Far West Rand gold mines', 'Springs coal field to Witbank coal field', 'Kliprand area to Kliprand area, W. Cape', 'Phalaborwa area, Limpopo to Vaalwater area, Limpopo', 'Swartland area to Swartruggens area', 'Kalkfontein dam area to Wolwefontein area', 'Cullinan area to Cullinan mine', 'Ikageleng area to Thakazele area', 'Carolusberg area to Carlsberg Ridge', 'Rite area to Nigel area', 'Sebokeng area to Maokeng area', 'Ga-Maplea mine to Ga-Mapela mine', 'Nylstroom area to Dullstroom area', 'Klerksdorp area to Kranskop area', 'Tikwana area to Dinokana area', 'Klerkdorp gold mines to Klerksdorp gold mines', 'Mphahlele area to Lephalale area', 'Cornelia area to Njonjela area', 'Robertson area to Robertson area, W. Cape', 'Williston area to Philipstown area', 'Mphakane area to Mphakane area, Limpopo', 'Evander gold mines to East Rand gold mines', 'Table View area to Table View area, W. Cape', 'Bhongweni area to Empangeni area', 'Swartkop area to Swartkop area, N. Cape', 'Kouebokkeveld area to Koue Bokkeveld area, W. Cape', 'Excelsior area to Nelspoort area', 'Prins Albert area to Prince Albert', 'Manzini area to Mtunzini area', 'Bellville area to Belleville area', 'Vosburg area to Fraserburg area', 'GaRamela area, Limpopo to GaMapela area, Limpopo', 'Gamoep area, W. Cape to Gamoep area, N. Cape', 'Simuelue, Indonesia to Simeulue, Indonesia', 'Prince Alfred Hamlet area, to Prince Alfred Hamlet, W. Cape', 'Rolfontein area to Wolwefontein area', 'Dewar area to Dendron area', 'Taung area to Tsineng area', 'Leuu-Gamka area to Leeu-Gamka area', 'Aldays area to Edashi area', 'GaMankopane area to GaRantlakane area', 'Kruger Nat. Park to Kruger national park', 'Groot - winterhoek area to Groot-Winterhoek area', 'Piet Retief area to Piet Retief, Mpumalanga', 'Venetia diamond mine to Venetia mine', 'Matatiele area to Maluti area', 'Kagga Kamma area to Kagga Kamma area, W. Cape', 'Witbank coal feild to Witbank coal field', 'Kakamas area to Kakamas area, N. Cape', 'Southern Botswana to Southern Iran', 'KaMhinga area to KwaMhlanga area', 'Lesotho Border to S.A - Lesotho Border', 'Vredendal area to Vredendal area, W. Cape', 'Balfour area, E. Cape to Bedfort area, East. Cape', 'Worcester area, W. Cape to Mortimer area, E. Cape', 'Wild coast area to Wild Coast area', 'Lime Acres area, N. Cape to De Aar area, N. Cape', 'Plooysburg area to Plooysburg area, N. Cape', 'Cathedral peak area to Cathedral Peak', 'Kgomo Kgomo area to Kgomo-Kgomo area', 'Western Indian Antarctic Ridge to Western Indian-Antarctic Ridge', 'Witsand area, N. Cape to Witsand area, W. Cape', 'Molatedi area to Maluti area', 'GaMalwane area to Vaalwater area', 'Carolina area to Harding area', 'Sutherland area to Soutpan area', 'Nicabar Islands, India region to Nicobar Islands, India region', 'Prince Edward Islands Region to Prince Edward Islands region', 'Klerksdorp gols mines to Klerksdorp gold mines', 'Montagu area to Montagu area, W. Cape', 'Ladybrand area to Ladybrand Region', 'Matjiesfontein area to Matjiesfontein, W. Cape', 'Scheepmoor area to Derdepoort area', "Libertador O'Higgens, Chile to Libertador O'Higgins, Chile", 'Senekal area to Seberia area', 'Matwabeng area to Mokgalwaneng area', 'Indian 0cean to Indian Ocean', 'Burgersdorp area to W. of Burgersdorp area', 'Sumba Region, Indonesia to Sumba region, Indonesia', 'Alldays area to Alldays area, Limpopo', 'Cullinan diamond mine to Cullinan mine', 'Ga-Selepe mine to Ga-Mapela mine', 'Danielskuil area to Danielskuil area, N. Cape', 'St Helena Bay to St Helena Bay, W. Cape', 'Far Wset Rand gold mines to Far West Rand gold mines', 'Carolasberg area to Fraserburg area', 'West of MacQuarie Island to West of Macquarie Island', 'Lake Malawi to Malawi', 'Hindu Kush region, Afhghanistan to Hindu Kush region, Afghanistan', 'Bela-Bela area, Limpopo to Bela-Bela, Limpopo', 'East London area to East London area, E. Cape', 'Roggeveldberge area, N.Cape to Roggeveldberge area, N. Cape', 'Sasolburg coal field to Sasolburg coal mines', 'Perth area to Northam area', 'Wolmaransstad area to Tarkastad area', 'Witbank coal fields to Witbank coal field', 'Sprinkana area to Dinokana area', 'Ohrigstad area to Tarkastad area', 'Diepsloot area to Derdepoort area', 'Schmidtsdrif area to Schmidtsdrif area, N. Cape', 'Deorham area to Northam area', 'Fraserburg area, N. Cape to Postmasburg area, N. Cape', 'Postmasburg area to Postmasburg area, N. Cape', 'Modikwe area to Morokweng area', 'Swartuggens area to Swartruggens area', 'Modimolle area to Fochville area', 'Ga-Mohlala area to KwaMhlanga area', 'Kermadec Islands, New Zealnad to Kermadec Islands region', 'Murraysburg area to Fraserburg area', 'Maseru - Lesotho to Maseru area, Lesotho', 'Sumbawa Region, Indonesia to Sumbawa region, Indonesia', 'South Sandwich Islands Region to South Sandwich Islands', 'Nicobar Islands, India Region to Nicobar Islands, India region', 'Cele area to Caledon area', 'Maebani area to Masameni area', 'Pella area to Nigel area', 'Offshore West coast of South Africa to Offshore West Coast of South Africa', 'Bushmanland area to Bushmanland, N. Cape', 'Rebone area to Rustdene area', 'Amdelbult mine to Amandelbult mine', 'Gamoep area to Ga-Thlose area', 'Far West and gold mines to Far West Rand gold mines', 'Wuppertal area to Wuppertal area, W. Cape', 'Vaaldam area to Vaal Dam area', 'Swaziland area to Swaziland border', 'Mogapeng area to Maipeng area', 'Ga-Makgopa area to GaMakibelo area', 'Far West rand gold mines to Far West Rand gold mines', 'Augrabies Falls area to Augrabies area', 'Prince Alfred Hamlet area, W. Cape to Prince Alfred Hamlet, W. Cape', 'Koringberg quarry to Piketberg quarry', 'Edenville area to Steynville area', 'Iran-Iraq Border to Iran-Iraq Border Region', 'Far west rand gold mines to Far West Rand gold mines', 'Bloemfontein area to Wolwefontein area', 'Cullinan platinum mine to Cullinan mine', 'Standerton area to Standerton Region', 'Cradock area to Caledon area', 'Manyatseng area to Maokeng area', 'Victoria West area to Victoria West area, N. Cape', 'Setateng area to Soutpan area', 'Lesotho area to Loxton area', 'Temba area to Timbavati area', 'Kepulauan Mentawai region, Inodnesia to Kepulauan Mentawai region, Indonesia', 'Camden area to Caledon area', 'Seabe area to Seberia area', 'GaMapala area to GaRankatlane area', 'Saron area to Soutpan area', 'Kepuluaun Mentawai region, Indonesia to Kepulauan Mentawai region, Indonesia', 'Far West Rand gold mines mines to Far West Rand gold mines', 'Far West gold mines to Far West Rand gold mines', 'Adendorp area to Dendron area', 'Phalaborwa area to Phalaborwa mine', 'Heilbron area to Dendron area', 'Matloding area to Madibong area', 'Keetmanshoop area to Kranskop area', 'Amersfoort area to Amersfoort rea', 'Zimbambwe to Zimbabwe', 'Pontdrif area to Pontdrif area,', 'Worcester area to Worcester quarry', 'Carnavon area to Carnavon area, N. Cape', 'Kenhardt area to Kenhardt area, N. Cape', 'Hantam area, W. Cape to Hantam area, N. Cape', 'Motlhabe area to Mametlhake area', 'Lac Kivu Region, DRC to Lac Kivu region, DRC', 'Rooiberg area to Rooiberg mine', 'Augrabies falls area to Augrabies area', 'Kruger National Park to Kruger national park', 'Groot-Winterhoek area, W. Cape to Groot-Winterhoek area', 'Bokkeveldberge to Bokkeveldberge, N. Cape', 'Ganyesa  area to Ganyesa', 'Westerberg area to Fraserburg area', 'Somerset East area to Somerset East area, E. Cape', 'Namakwaland area to Namakwaland area, N. Cape', 'Hantam area to Hantam area, N. Cape', 'Ulundi area to Maluti area', 'Copperton area to Loxton area', 'Bouvet Island Region to Bouvet Island region', 'Nias Region, Indonesia to Nias region, Indonesia', 'Andreanof Islands, Aleutian Is., Alaska to Andreanof Islands, Aleutian Islands, Alaska', 'Groothoek mine to Bothashoek mine', 'Vennetia mine to Venetia mine', 'Yzerfontein area to Wolwefontein area', 'Queenstown area to Queenstown area, E. Cape', 'Dibeng area to Tsineng area', 'Phokwane area to Mokgalwaneng area', 'Steinkopf area to Steinkopf area, N. Cape', 'Sheepmoor area to Derdepoort area', 'Ficksburg area to Fraserburg area', 'Far West Rand gold mine to Far West Rand gold mines', 'Matjiesfontein area, W. Cape to Matjiesfontein, W. Cape', ': Southern Mozambique to Southern Zambia', 'Stofberg area to Seberia area', 'Agenneys area to Ngwenwane area', 'Osizweni area to Ntsikeni area', 'Freschhoek area, W. Cape to Franschoek area, W. Cape', 'Groot Winterhoek area to Groot-Winterhoek area', 'Hopetown area to Loxton area', 'De Doorns area to De Doorns area, W. Cape', 'Witbank coal Field to Witbank coal field', 'Olifantshoek area to Olifantshoek area, N. Cape', 'Rustenburg  area to Rustdene area', 'Kagiso area to Kranskop area', 'Aggeneys area to Aggeneys mine', ': Witbank Coal field to Witbank coal field', 'Mount Fletcher to Mount Fletcher area', ':  Witbank coal field to Witbank coal field', 'Northern Mid Atlantic Ridge to Norhern Mid Atlantic Ridge', 'Maapea area to Maipeng area', 'Tweebontein area to Wolwefontein area', 'Memel area to Keimoes area', 'Madikwe area to Madikgetla area', ': Koffiefontein area to Koffiefontein area', 'Redelinghuys area to Redelinghuys area, W. Cape', 'Botswana area to Botswana', 'Mataleng area to Maokeng area', 'Ngololeni area to Noenieput area', ':  Klerksdorp gold mines to Klerksdorp gold mines', 'Qudeni area to Rustdene area', 'Far West Rand gold  mines to Far West Rand gold mines', 'Tulbagh area to Tulbagh area, W. Cape', 'Boesmansland area to Boesmanland, N. Cape', 'Sishen manganse field to Sishen manganese field', 'Amandelboom area to Amandelboom area, N. Cape', 'Bethulie area to Bothaville area', 'Bitterfontein area to Koffiefontein area', 'Venetias mine to Venetia mine', 'Mangaung area to Maokeng area', 'Rooifontein area, N. Cape to Upington area, N. Cape', 'GaMothele area to GaMothele area, Limpopo', 'Knersvlakte, W. Cape to Knersvlakte area, W. Cape', 'Mozambique Channel to Mozambique', 'Kalkfontein Dam area to Wolwefontein area', 'Augarbies area to Augrabies area', 'Sada area to Nkandla area', 'Oranjeville area to Merweville area', 'Indian ocean to Indian Ocean', 'Driefontein to Driefontein colliery', 'South of Java, Indoneasia to South of Java, Indonesia', 'Ga-Mampa mine to Ga-Mapela mine', 'Philippolis area to Philipstown area', 'S.W. Zimbabwe to Zimbabwe', 'Frankfort area to Arnot area', 'Ditloung area to Boitumelong area', 'Bushveld Platinum mines to Bushveld platinum mines', 'Fish Hoek area to Boitshoko area', 'GaMasemola area to GaMasemolo area', 'Zambia to Namibia', 'GaRamela area to GaRantlakane area', 'Nababeep area to Nababeep area, N. Cape', 'Tholeni area to Theunissen area', 'GaMapela area to GaMasemolo area', 'Sishen Manganese mine to Sishen manganese field', 'Grootegeluk mine to Grootegeluk coal mine', 'Groot-Swartberge area to Groot Swartberge', 'Pofadder area, N. Cape to Soetwater area, N. Cape', 'Tshilwana area to Tshakhuma area', 'Bay area to Doorn Bay area', 'Ramohlakoana area to GaRantlakane area', 'Warden area to Harding area', 'Boesmanland Area - N. Cape to Boesmanland, N. Cape', 'Renostervlei area to Rhenosterkraal area', 'Roedtan area to Soutpan area', 'Winberg area to Tsineng area', 'Aurora area to Arnot area', 'Qibing area to Maipeing area', 'Derby area to Dendron area', 'Far west Rand gold mines to Far West Rand gold mines', 'Winterberge area to Winterberge area, E. Cape', 'Klawer area to Vaalwater area', 'Mahwelereng area to Mahwelereng area, Limpopo', 'Prince Albert area to Prince Albert', 'Sasolburg coal mine to Sasolburg coal mines', 'Nqabeni area to Ntsikeni area', 'Amandelbult area to Amandelbult mine', 'Augrabies rea to Augrabies area', 'Mahareng area to Maokeng area', 'Great Karoo area to Great Karoo area, N. Cape', 'Petrusburg area to Fraserburg area', 'Bergville area to Hertzogville area', 'South Sandwich Islands region to South Sandwich Islands', 'Afghanistan-Tajikistan Border Region to Afghanistan-Tajikistan border region', 'Free gold mines to Free State gold mines', 'Rawsonville area to Rawsonville area, W. Cape', 'Mozambique channe to Mozambique', 'Myanmar-India Border Region to Myanmar-India border region', 'Ottosdal area to Ottoshoop area', 'Sasolburg coal mine       ` to Sasolburg coal mines', 'Clarens area to Caledon area', 'Kagga Kamma area,  W. Cape to Kagga Kamma area, W. Cape', 'Vaalwate area to Vaalwater area', 'Lake Kariba Zimbabwe to Lake Kariba, Zimbabwe', 'Ganspan area to Masameni area', 'Steelpooort mines to Steelpoort mines', 'Witbank Coal field to Witbank coal field', 'venetia mine to Venetia mine', 'West of Potchesftroom to West of Potchefstroom', 'Cathedral Peak area` to Cathedral Peak', 'Kroonstad area to Kokstad Area', 'Porterville area to Petrusville area', 'South Sandwich Island region to South Sandwich Islands', 'Hoopstad area to Kokstad Area', 'Cwebeni area to Seberia area', 'Steelpoort mine to Steelpoort mines', 'Batleng area to Maokeng area', 'Aggenneys area to Aggeneys mine', 'Kathu area to Matsulu area', 'Keimos area to Keimoes area', 'Dingleton area to Dinokana area', 'Namqualand area to Namaqualand', 'Sultanaoord area to Soutpan area', 'Lorriesfontein area to Koffiefontein area', 'Loeriesfontein area, N. Cape to Gordonia area, N. Cape', 'GaMolepo area to GaMasemolo area', 'Betleng area to Itsoseng area', 'Comores Island region to Bouvet Island region', ': Witbank coal field to Witbank coal field', 'highveld coal field to Highveld coal field', 'Delportshoop area to Ottoshoop area', 'Potfontein area to Wolwefontein area', 'Nous area to Volksrus area', 'Koue Bokkeveld area to Koue Bokkeveld area, W. Cape', 'Sishen manganese mine to Sishen manganese field', ': Far West Rand gold mines to Far West Rand gold mines', 'Setlagole area to Lephalale area', 'Richmond area to Richmond area, N. Cape', 'Kokstad area to Kokstad Area', 'Southern Mid Atlantic Ridge to Southern Mid-Atlantic Ridge', 'Luxolweni area to Wolwefontein area', 'Louistrichardt area to Louis Trichardt area', 'Meyerton area to Dendron area', 'Hammanskraal area to Hammanskraal mine', 'Fauresmith area to Harrismith area', 'Bio-Bio, Chile to Bio Bio, Chile', 'Gamatlala area to GaRantlakane area', 'Mozambique border to Mozambique', 'Carletonville area to Cedarville area', 'Tigane area to Tsineng area', 'Ga-Makgopa mine to Ga-Mapela mine', 'Nuwerus area to Nuwerus area, W. Cape', 'Griekwastad area to Tarkastad area', 'Morreesburg area to Fraserburg area', 'Mokolo dam area to Mokolo dam area, Limpopo', 'Magashoa area to Magoebaskloof area', 'Sneeuberge area to Sneeuberge area, E. Cape', 'Boesmanland area, N. Cape to Boesmanland, N. Cape', 'Tshibeng mine to Letseng mine', 'Bushmanland area, N. Cape to Bushmanland, N. Cape', 'Hluvukani area to Tutuka area', 'Far West Rand gold field to Far West Rand gold mines', 'Prieska area to Prieska area, N. Cape', 'Grootvloer area to Grootvloer area, N. Cape', 'Jagersfontein area to Wolwefontein area', 'Moebani area to Mkomazi area', 'Vryburg area to Fraserburg area', 'Marble Hall Region to Marble Hall area', "Mohale'shoek, Lesotho to Mohale's hoek, Lesotho", 'Madipelesa area to Madikgetla area', 'Namaqualand area to Namaqualand', 'Koppies area to Keimoes area', 'Pilanesberg Nat. Park to Pilanesberg National Park', 'Ventersburg area to Fraserburg area', 'Far West Rand Gold Mines to Far West Rand gold mines', 'Viljoenskroon area to Viljoenskroon', 'Rustenburg Area to Rustenburg area, N. West', 'Indien ocean to Indian Ocean', 'Maganeng area to Mokgalwaneng area', 'Witbank cold field to Witbank coal field', 'Greylingstad area to Greylingstad', 'Van Wyksvlei area to Van Wyksvlei area, N. Cape', 'Maake area to Maokeng area', 'Prince Alfred Area to Prince Albert', 'Indian Ridge to Mid-Indian Ridge', 'Mabotsha area to Northam area', 'Bushveld platinim mines to Bushveld platinum mines', 'Piet Retief-Mpumalanga to Piet Retief, Mpumalanga', 'Letaba area to Lephalale area', 'Phalabora mine to Phalaborwa mine', 'Kuruman area to Kuruman area, N. Cape', 'Edenburg area to Fraserburg area', 'Loeriesfontein area to Koffiefontein area', 'Roggeveldberge, N. Cape to Roggeveldberge area, N. Cape', 'Centurion area to Dendron area', 'Ngweding mine to Jwaneng mine', 'Bio-bio, Chile to Bio Bio, Chile', 'Namibia border to SA-Namibia border', 'Rat Islands, Aleution Islands to Rat Islands, Aleutian Islands', 'GaMarishane area to GaMasemolo area', 'Barkly West area to Beaufort West area', 'Makomereng area to Maokeng area', 'Barbeton area to Caledon area', 'Marble Hall mine to Marble Hall area', 'Holfontein quarry to Olifantsfontein quarry', 'Tarapaca, chile to Tarapaca, Chile', 'Groote geluk mine to Grootegeluk coal mine', 'Vorstershoop area to Ottoshoop area', 'Ga-Mampuru mine to Ga-Mapela mine', 'Ga-Tlhose area to Ga-Thlose area', 'Boesmanland to Boesmanland, N. Cape', ': East Rand gold mines to East Rand gold mines', 'Boesmanland - N. Cape to Boesmanland, N. Cape', 'Dealesville area to Deneysville area', 'Prince Alfred Hamlet area to Prince Alfred Hamlet, W. Cape', 'Masobe area to Maokeng area', 'Kruger park to Kruger national park', 'Amendelbult mine to Amandelbult mine', 'SA Botswana border to Botswana/Namibia border', 'Rustenburg area to Rustdene area', 'Nuweveld Berge to Nuweveldberge', 'Marogotholong area to Mogohlwaneng area', 'Klerksdorp gold mine to Klerksdorp gold mines', 'Beeshoek area to Beeshoek area, N. Cape', 'Mid Indian Ridge to Mid-Indian Ridge', 'Brandvlei area to Brandvlei area, N. Cape', 'Mozambique channel to Mozambique', 'Amalanga area to KwaMhlanga area', 'GaMasemdla area to GaMasemolo area', 'Niekerkshoop area to Niekerkshoop area, N. Cape', 'Hlobane area to Hobhouse area', 'Zimbabwe border to Zimbabwe, SA border', ': Klerksdorp gold mines to Klerksdorp gold mines', 'Groblershoop area to Groblersdal area', 'Thabazimbi area to Thabazimbi area, N. West', 'Warmbath area to Harrismith area', 'Andreanof Islands, Aleution Islands, Alaska to Andreanof Islands, Aleutian Islands, Alaska', 'Okavango Delta -Botswana to Okavango Delta, Botswana', 'Ntshongweni area to Ntsikeni area', 'Mlungisi area to Mtunzini area', 'Soutpansberg, Limpopo to Soutpansberg area, Limpopo', 'Letseng diamond mine to Letseng mine', 'GaMatlala area to GaRantlakane area', 'Filabusi - Zimbabwe to Filabusi, Zimbabwe', 'Keiskammahoek area to Kieskammahoek area, E. Cape', 'Nuweveld berge to Nuweveldberge', 'GaMabalebele area to GaMakibelo area', 'Richterveld Nat. Reserve to Richterveld Nature Reserve', 'Dedeben area to Dendron area', 'Rat Islands, Aluetian Islands to Rat Islands, Aleutian Islands', 'Ga-Mapela to Ga-Mapela mine', 'Tweefontein area to Wolwefontein area', 'Saldanha Bay area to Saldanha bay', 'Bothaville to Bothaville area', 'Fisantkraal mine to Hammanskraal mine', 'Madipalesa area to Maipeng area', 'Touwsrivier area to Touwsrivier area, W. Cape', 'Richtersveld -  Namibia to Richtersveld, Namibia', 'Mzongwana area to Mokgalwaneng area', 'Fort Beaufort area to Beaufort West area', 'Central Rand gold mine to Central Rand gold mines', 'Sada area, East. Cape to Sada area, E. Cape', 'E. Transvaal coal field to Ermelo coal field', 'Witsand area to eZitandini area', 'Umtata area to Etwatwa area', 'Somerset-east area to Somerset East area, E. Cape', 'Saldanha area to Saldanha bay', 'Khutsong area to Itsoseng area', 'Upinton area to Dinokana area', 'Gariep Dam area to Gariep Dam', 'Ntsikeni Nature Reserve to Ntsikeni Wildlife reserve', 'St Helena bay area to St Helena Bay, W. Cape', 'Springfontein area to Koffiefontein area', 'Ammerville area to Merweville area', 'Komaggas area to Komaggas area, N. Cape', 'Fox Islands, Aluetion Islands to Fox Islands, Aleutian Islands', 'Kepulauan Memtawai region, Indonesia to Kepulauan Mentawai region, Indonesia', 'Pongola area to Nongoma area', 'Lime Acres area to Lime Acres mine', 'Makgake area to Maokeng area', 'Ugie area to Nigel area', 'Maromeni area to Masameni area', 'Fateng area to Maokeng area', 'Tzaneen area to Tsineng area', 'Phiritona area to Philipstown area', 'Bredasdorp area to Humansdorp area', 'Masilo area to Matsulu area', 'Gariep dam area to Gariep Dam', 'Musina area to Mussina area', 'Ngodwana area to Nkowakowa area', 'Phomolong area to Boitumelong area', 'Southern Namibia to Southern Zambia', 'Welbedagtdam area to Welbedagtdam', 'Jacobsdal area to Groblersdal area', 'Kaiingveld  area, N. Cape to Kaiingveld area, N. Cape', 'Central Lesotho to Central Karoo', 'Strydenburg area to Fraserburg area', 'Democratic Republic of the congo to Democratic Republic of the Congo', 'Soutpansberg area to Soutpan area', 'Reivilo area to Rietkuil area', 'Southwest of sumatra, Indonesia to Southwest of Sumatra, Indonesia', 'Gariep dam to Gariep Dam', 'Colenso area to Caledon area', 'Myanmar-India Border region to Myanmar-India border region', 'Stella area to Steynville area', 'Mashaeng area to Masameni area', 'Pudiyakgopa area to Pudiyakgomo area', 'Augrabies to Augrabies area', 'Kleinsee area to Keimoes area', 'Ratanda area to Nkandla area', 'Murrraysburg area to Fraserburg area', 'Colesberg area to Colesberg', 'Newcastle region to Newcastle area', 'Calitzdorp area to Caledon area', 'Tankwakaroo area to Tankwa Karoo area, N. Cape', 'Mokolo Dam area to Mokolo dam area, Limpopo', 'southern Mozambique to Southern Zambia', 'Kendrew area to Dendron area', 'Chile-Argentina Border Region to S. Chile-Argentina Border Region', 'Betlehem area to Mametlhake area', 'Tlapeng area to Maipeng area', 'Kalbasfontein area to Wolwefontein area', 'Cooperdale area to Cedarville area', 'Mvumane area to Masinyusane area', 'Matleding area to Maipeing area', 'Bredasdrop area to Kranskop area', 'Kaiingveld area to Kaiingveld area, N. Cape', 'Reddersburg area to Fraserburg area', 'Vredendal area, N. Cape to Vredendal area, W. Cape', 'Theunessin area to Theunissen area', 'Swartwater area to Vaalwater area', 'Mpunzana area to Mtunzini area', 'East Rand Gold Mines to East Rand gold mines', 'Kagga kamma area to Kagga Kamma area, W. Cape', 'Thaba nchu area to Thaba Nchu area', 'Marble hall area to Marble Hall area', 'Welgevonden area to Wolwefontein area', 'Korinte area to Koffiefontein area', 'Marapyane area to Masinyusane area', 'Ulco area to Clocolan area', 'Kepulauan Taluad, Indonesia to Kepulauan Talaud, Indonesia', 'Karkams area to Tarkastad area', 'St.Helena Bay area to St Helena Bay, W. Cape', 'Ga-Rankuwa area to GaRankatlane area', 'Namakgale area to Thakazele area', 'Bushmanland to Bushmanland, N. Cape', 'Phillipstown area to Philipstown area', 'Nelspruit area to Nelspoort area', 'Lake Tanganyika to Lake Tanganyika Region', 'Witbank coal filed to Witbank coal field', 'Potlake area to Mametlhake area', 'Palabora mine to Phalaborwa mine', 'Ceres area to Keimoes area', 'Lake Tanganyika, Tanzania to Lake Tanganyika region, Tanzania', 'Groot Winterhoek Park to Groot-Winterhoek area', 'Trompsburg area to Fraserburg area', 'Hlohlolwane area to Clocolan area', 'Citrusdal area to Citrusdal area, W. Cape', 'Mokgareng area to Maokeng area', 'Malmesbury area to Malmesbury area, W. Cape', 'Amsterdam area to Amsterdam Region', 'Sabie area to Augrabies area', 'Democratric Republic of the Congo to Democratic Republic of the Congo', 'Maclear area to Marble Hall area', 'Ottoshoop Area to Ottoshoop area', 'Kalkfontein area to Wolwefontein area', 'Warrenton area to Warrenton area, N. Cape', 'Poffader area to Pofadder area', 'Atlantic Ocean to Atlantic ocean', 'Driefontein colierry to Driefontein colliery', 'Northern Botswana to Northern Iran', 'Owen Fracture Zone Region to Owen Fracture Zone region', 'Gordonia area to Gordonia area, N. Cape', 'Deelfontein area, N. Cape to Dedeben area, N. Cape', 'Brandfort area to Arnot area', 'Matroosberg area to Matroosberg area, W. Cape', 'Prince Alfred Harmlet to Prince Alfred Hamlet, W. Cape', 'Huib-Hoch plateau Namibia to Huib-Hoch Plateau Namibia', 'Letswatla area to Etwatwa area', 'Mbombo area to Mkhombo area', 'Keimoes to Keimoes area', 'Ionian Sea to Indian Ocean', 'Edendale area to Cedarville area', 'Cathedral Peak area to Cathedral Peak', 'Tsilwana area to Tsineng area', 'Sasol coal field to Ermelo coal field', 'St. Helena Bay area to St Helena Bay, W. Cape', 'Kwaggafontein area to Wolwefontein area', 'West chile Rise to West Chile Rise', 'Geysdorp area to Humansdorp area', 'Bronkhorstspuit area to Bronkhorstspruit area', 'Groot-swartberge area to Groot Swartberge', 'Lesotho border to S.A - Lesotho Border', 'Moteti area to Maluti area', 'Mokopane mine to Mothabe mine', 'Kruger Nat. park to Kruger national park', 'East Rand gold mine to East Rand gold mines', 'Pofadder Region to Pofadder area', 'Piketberg area to Piketberg quarry', 'Andreanof Islands, Aluetian Islands to Andreanof Islands, Aleutian Islands', 'MacQuarie Islands region to Macquarie Island Region', 'Domboni area to Mkhombo area', 'Lake Kariba area to Lake Kariba, Zambia', 'Southern Zimbabwe to Southern Zambia', 'Botswana border to Botswana/Namibia border', 'Mokopane area to Mookgopong area', 'Umzimkulu area to Umzimkhulu area', 'Higveld coal field to Highveld coal field', 'Kraankuil area to Kranskop area', 'Dundee area to Rustdene area', 'Witfontein quarry to Olifantsfontein quarry', "Pearston area to Pearson's Hunt area", 'Grootgeluk coal mine to Grootegeluk coal mine', 'Ritchie area to Smithfield area', 'Swartreggens area to Swartruggens area', 'Off the coast of Bio-Bio, chile to Off the coast of Bio-Bio, Chile', 'Phahameng area to Tlhakgameng area', 'Prince Edward Island Region to Prince Edward Islands region', 'Bio-Bio, chile to Bio Bio, Chile', 'Offshore Bio-bio, Chile to Offshore Bio-Bio, Chile', 'Mookane area to Morokweng area', 'Golden Gates Highlands to Golden Gate Highlands National Park', 'Aggeneyes area to Aggeneys mine', 'Mabeskraal area to Mabeskraal area, N. West', 'Williston area, N. Cape to Wellington area, W. Cape', 'Kepulauan Mentawai Region, Indonesia to Kepulauan Mentawai region, Indonesia', 'Grootdrink area to Groot-Winterhoek area', 'Pieketberg area to Piketberg quarry', 'Sishen mine area to Sishen mine area, N. Cape', 'Free state gold mines to Free State gold mines', 'Hanover area to Humansdorp area', ': Free State gold mines to Free State gold mines', 'Knersvlakte area to Knersvlakte area, W. Cape', 'Swaartrugens area to Swartruggens area', 'Seaview area to Hazyview area', 'Prince Edward Islands to Prince Edward Islands region', 'Comores region to Comoros region', 'Polokwane area to Polokwane area, Limpopo', 'Mogwase area to Mokgalwaneng area', 'Lake Edward region, Dem. Rep. of the Congo to Lake Edward region, Democratic Republic of the Congo', 'Villiers area to Villa Nora area', 'Leeu-Gasmka area to Leeu-Gamka area', 'Ntwane area to Ngwenwane area', 'Augrabies  area to Augrabies area', 'Western Indian-Antartic Ridge to Western Indian-Antarctic Ridge', 'De Aar area to De Aar area, N. Cape', "Mohales Hoek, Lesotho to Mohale's hoek, Lesotho", 'Mammutla area to Mametlhake area', 'Boesmanland area to Boesmanland, N. Cape', 'Ganyesa area to Ganyesa', 'Sediba area to Seberia area', 'Mafikeng area to Maokeng area', 'Rammulotsi area to Ratswini area', 'Klein-Winterhoekberge area to Klein_Winterhoekberge area, E. Cape', 'Hindu Kush Region, Afghanistan to Hindu Kush region, Afghanistan', 'Sterkspruit area to Hectorspruit area', 'Upington area to Upington area, N. Cape', 'Dinopeng area to Dinokaneng area', 'KwaThandeka area to KwaMhlanga area', 'Carnarvon area to Carnarvon area, N. Cape', 'Leeu Gamka area to Leeu-Gamka area', 'Boesmanland Area to Boesmanland, N. Cape', 'Kirkwood area to Kirkwood area, E. Cape', 'South Indian Ocean to Indian Ocean', 'Offshore East Coast of south Africa to Offshore East Coast of South Africa', 'Steelpoort area to Nelspoort area', 'Offshore bio-Bio, Chile to Offshore Bio-Bio, Chile', 'Madadeni area to Masameni area', 'Fisankraal mine to Hammanskraal mine', 'Zamani area to Zigagayi area', 'GaRankatlane  area to GaRankatlane area', 'Boland area to Clocolan area', 'Zembeni area to Seberia area', 'Bergsig area to Blyvoortsig area', 'Southy area to Southey area', 'Soutpansberge area, Limpopo to Soutpansberg area, Limpopo', 'Reagile area to Cedarville area', 'Ventersdorp area to Humansdorp area', 'Lothair area to Northam area', 'Murraysburg  area to Fraserburg area', 'EThembeni area to Empangeni area', 'Bushveld platinum nines to Bushveld platinum mines', 'Riebeek-Oos area to Riebeek-Oos area, E. Cape', 'Roggeveldberge area to Roggeveldberge area, N. Cape', 'Acornhoek area to Arnot area', 'Nuweveldberge area to Nuweveldberge', 'Ga Mapela mine to Ga-Mapela mine', 'Tlokweng area to Morokweng area', 'Rooifontein area to Koffiefontein area', 'De Hoek area to De Hoek quarry', 'Garies area to Augrabies area', 'Weenen area to Van Reenen area', 'Hartswater area to Vaalwater area', 'Morekweng area to Morokweng area', 'Strand area to Soutpan area', 'Namakwaland to Namaqualand', 'Saulspoort area to Maselspoort area', 'Off coast Cape St. Lucia to Off coast Cape St Lucia', 'Mauritius - Reunion region to Mauritius/Reunion region', 'Amandelbult Mine to Amandelbult mine', 'Beaufort west area to Beaufort West area', 'Driefontein area to Koffiefontein area', 'Balasi area to Maluti area', 'Klaarstroom area to Wakkerstroom area', 'Bultfontein area to Wolwefontein area', 'sishen Manganese mines to Sishen manganese field', 'Britstown area to Britstown area, N. Cape', 'Anysberg area to Anysberg area, W. Cape', 'Oudtshoorn area to Oudtshroon area', 'Sakrivier area to Sakrivier area, N. Cape', 'Offsore Bio-Bio, Chile to Offshore Bio-Bio, Chile', 'Waterberge area to Waterberge area, Limpopo', 'Prince Alfred Hamlet to Prince Alfred Hamlet, W. Cape']

    nomatch.sort()
    correction.sort()

    nomatchstr = r''
    for i in nomatch:
        nomatchstr += i + '\n'

    correctionstr = r''
    for i in correction:
        correctionstr += i + '\n'


    with open(r'c:\workdata\nomatch.txt', 'w') as fno:
        fno.write(nomatchstr)

    with open(r'c:\workdata\correction.txt', 'w') as fno:
        fno.write(correctionstr)


    breakpoint()



if __name__ == "__main__":
    main()
