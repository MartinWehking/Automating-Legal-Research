import logging

import re
import requests
import unidecode
from bs4 import BeautifulSoup

base = "https://www.gesetze-im-internet.de/Teilliste_translations.html"

urls = {
    "https://www.gesetze-im-internet.de/englisch_abgg/englisch_abgg.html",
    "https://www.gesetze-im-internet.de/englisch_agg/englisch_agg.html",
    "https://www.gesetze-im-internet.de/englisch_amg/englisch_amg.html",
    "https://www.gesetze-im-internet.de/englisch_antidopg/englisch_antidopg.html",
    "https://www.gesetze-im-internet.de/englisch_ao/englisch_ao.html",
    "https://www.gesetze-im-internet.de/englisch_arbmedvv/englisch_arbmedvv.html",
    "https://www.gesetze-im-internet.de/englisch_arbschg/englisch_arbschg.html",
    "https://www.gesetze-im-internet.de/englisch_arbst_ttv/englisch_arbst_ttv.html",
    "https://www.gesetze-im-internet.de/englisch_asig/englisch_asig.html",
    "https://www.gesetze-im-internet.de/englisch_asylvfg/englisch_asylvfg.html",
    "https://www.gesetze-im-internet.de/englisch_aufenthg/englisch_aufenthg.html",
    "https://www.gesetze-im-internet.de/englisch_aug/englisch_aug.html",
    "https://www.gesetze-im-internet.de/englisch_awaffv/englisch_awaffv.html",
    "https://www.gesetze-im-internet.de/englisch_awg/englisch_awg.html",
    "https://www.gesetze-im-internet.de/englisch_awv/englisch_awv.html",
    "https://www.gesetze-im-internet.de/englisch_bbergg/englisch_bbergg.html",
    "https://www.gesetze-im-internet.de/englisch_bdsg/englisch_bdsg.html",
    "https://www.gesetze-im-internet.de/englisch_berathig/englisch_berathig.html",
    "https://www.gesetze-im-internet.de/englisch_betrvg/englisch_betrvg.html",
    "https://www.gesetze-im-internet.de/englisch_bfdg/englisch_bfdg.html",
    "https://www.gesetze-im-internet.de/englisch_bgb/englisch_bgb.html",
    "https://www.gesetze-im-internet.de/englisch_bgbeg/englisch_bgbeg.html",
    "https://www.gesetze-im-internet.de/englisch_bgleig/englisch_bgleig.html",
    "https://www.gesetze-im-internet.de/englisch_bgrembg/englisch_bgrembg.html",
    "https://www.gesetze-im-internet.de/englisch_biostoffv/englisch_biostoffv.html",
    "https://www.gesetze-im-internet.de/englisch_bmg/englisch_bmg.html",
    "https://www.gesetze-im-internet.de/englisch_bsig/englisch_bsig.html",
    "https://www.gesetze-im-internet.de/englisch_bverfgg/englisch_bverfgg.html",
    "https://www.gesetze-im-internet.de/englisch_geschmmg/englisch_geschmmg.html",
    "https://www.gesetze-im-internet.de/englisch_drig/englisch_drig.html",
    "https://www.gesetze-im-internet.de/englisch_egovg/englisch_egovg.html",
    "https://www.gesetze-im-internet.de/englisch_erws_ag/englisch_erws_ag.html",
    "https://www.gesetze-im-internet.de/englisch_euabgg/englisch_euabgg.html",
    "https://www.gesetze-im-internet.de/englisch_euzbbg/englisch_euzbbg.html",
    "https://www.gesetze-im-internet.de/englisch_famfg/englisch_famfg.html",
    "https://www.gesetze-im-internet.de/englisch_feuerschstg/englisch_feuerschstg.html",
    "https://www.gesetze-im-internet.de/englisch_flugdag/englisch_flugdag.html",
    "https://www.gesetze-im-internet.de/englisch_freiz_gg_eu/englisch_freiz_gg_eu.html",
    "https://www.gesetze-im-internet.de/englisch_gg/englisch_gg.html",
    "https://www.gesetze-im-internet.de/englisch_gmbhg/englisch_gmbhg.html",
    "https://www.gesetze-im-internet.de/englisch_gvg/englisch_gvg.html",
    "https://www.gesetze-im-internet.de/englisch_gwb/englisch_gwb.html",
    "https://www.gesetze-im-internet.de/englisch_hgb/englisch_hgb.html",
    "https://www.gesetze-im-internet.de/englisch_ifg/englisch_ifg.html",
    "https://www.gesetze-im-internet.de/englisch_inso/englisch_inso.html",
    "https://www.gesetze-im-internet.de/englisch_eginso/englisch_eginso.html",
    "https://www.gesetze-im-internet.de/englisch_intfamrvg/englisch_intfamrvg.html",
    "https://www.gesetze-im-internet.de/englisch_intvg/englisch_intvg.html",
    "https://www.gesetze-im-internet.de/englisch_irg/englisch_irg.html",
    "https://www.gesetze-im-internet.de/englisch_jfdg/englisch_jfdg.html",
    "https://www.gesetze-im-internet.de/englisch_jgg/englisch_jgg.html",
    "https://www.gesetze-im-internet.de/englisch_kapmug/englisch_kapmug.html",
    "https://www.gesetze-im-internet.de/englisch_kgsg/englisch_kgsg.html",
    "https://www.gesetze-im-internet.de/englisch_lasthandhabv/englisch_lasthandhabv.html",
    "https://www.gesetze-im-internet.de/englisch_lpartg/englisch_lpartg.html",
    "https://www.gesetze-im-internet.de/englisch_marimedv/englisch_marimedv.html",
    "https://www.gesetze-im-internet.de/englisch_markeng/englisch_markeng.html",
    "https://www.gesetze-im-internet.de/englisch_mediationsg/englisch_mediationsg.html",
    "https://www.gesetze-im-internet.de/englisch_milog/englisch_milog.html",
    "https://www.gesetze-im-internet.de/englisch_nwrg/englisch_nwrg.html",
    "https://www.gesetze-im-internet.de/englisch_oeg/englisch_oeg.html",
    "https://www.gesetze-im-internet.de/englisch_offshore-arbzv/englisch_offshore-arbzv.html",
    "https://www.gesetze-im-internet.de/englisch_owig/englisch_owig.html",
    "https://www.gesetze-im-internet.de/englisch_partgg/englisch_partgg.html",
    "https://www.gesetze-im-internet.de/englisch_pa_g/englisch_pa_g.html",
    "https://www.gesetze-im-internet.de/englisch_patg/englisch_patg.html",
    "https://www.gesetze-im-internet.de/englisch_pauswg/englisch_pauswg.html",
    "https://www.gesetze-im-internet.de/englisch_prodhaftg/englisch_prodhaftg.html",
    "https://www.gesetze-im-internet.de/englisch_prodsg/englisch_prodsg.html",
    "https://www.gesetze-im-internet.de/englisch_rdg/englisch_rdg.html",
    "https://www.gesetze-im-internet.de/englisch_rpflg/englisch_rpflg.html",
    "https://www.gesetze-im-internet.de/englisch_rvg/englisch_rvg.html",
    "https://www.gesetze-im-internet.de/englisch_schbesv/englisch_schbesv.html",
    "https://www.gesetze-im-internet.de/englisch_seearbg/englisch_seearbg.html",
    "https://www.gesetze-im-internet.de/englisch_seearb_v/englisch_seearb_v.html",
    "https://www.gesetze-im-internet.de/englisch_see-arbznv/englisch_see-arbznv.html",
    "https://www.gesetze-im-internet.de/englisch_see-bav/englisch_see-bav.html",
    "https://www.gesetze-im-internet.de/englisch_see-bv/englisch_see-bv.html",
    "https://www.gesetze-im-internet.de/englisch_seeunterkunftsv/englisch_seeunterkunftsv.html",
    "https://www.gesetze-im-internet.de/englisch_rustag/englisch_rustag.html",
    "https://www.gesetze-im-internet.de/englisch_stgb/englisch_stgb.html",
    "https://www.gesetze-im-internet.de/englisch_stpo/englisch_stpo.html",
    "https://www.gesetze-im-internet.de/englisch_stvollzg/englisch_stvollzg.html",
    "https://www.gesetze-im-internet.de/englisch_umwelthg/englisch_umwelthg.html",
    "https://www.gesetze-im-internet.de/englisch_umwg/englisch_umwg.html",
    "https://www.gesetze-im-internet.de/englisch_urhg/englisch_urhg.html",
    "https://www.gesetze-im-internet.de/englisch_uwg/englisch_uwg.html",
    "https://www.gesetze-im-internet.de/englisch_versstg/englisch_versstg.html",
    "https://www.gesetze-im-internet.de/englisch_vgg/englisch_vgg.html",
    "https://www.gesetze-im-internet.de/englisch_vsbg/englisch_vsbg.html",
    "https://www.gesetze-im-internet.de/englisch_vvg/englisch_vvg.html",
    "https://www.gesetze-im-internet.de/englisch_vwgo/englisch_vwgo.html",
    "https://www.gesetze-im-internet.de/englisch_waffg/englisch_waffg.html",
    "https://www.gesetze-im-internet.de/englisch_woeigg/englisch_woeigg.html",
    "https://www.gesetze-im-internet.de/englisch_zpo/englisch_zpo.html",
    "https://www.gesetze-im-internet.de/englisch_zvg/englisch_zvg.html"}


def getTexts(urls):
    texts = []
    for u in urls:
        logging.info("Getting %s", u)

        lawcode = u.split('/')[3].split('_')[1]

        r = requests.get(u)
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove all links
        for a in soup.find_all('a'):
            a.decompose()

        # Jump to disclaimer
        siblings = soup.find("hr").next_siblings

        division_id = ""
        division_topic = ""

        section_id = ""
        section_topic = ""
        paragraph_nr = '(0)'
        for s in siblings:
            if s.has_attr('style') and len(s.contents) == 3:
                id = unidecode.unidecode(s.contents[0])
                m = re.match("Division \d", str(id))
                if m is not None:
                    division_id = str(id)
                    division_topic = s.contents[2]
                else:
                    section_id = id
                    section_topic = s.contents[2]
                paragraph_nr = '(0)'
            elif s.has_attr('style'):
                continue
            else:
                text = unidecode.unidecode(s.get_text()).strip()

                m = re.match('(\(\d+\))(.*)', text)
                if m is not None:
                    paragraph_nr = m.group(1)
                    text = m.group(2)

                ident = lawcode + " " + str(section_id) + " " + paragraph_nr
                tuple = (ident, section_topic, division_id + ": " + division_topic, text)
                if section_topic != "(no longer applicable)":
                    texts.append(tuple)

    return texts


if __name__ == '__main__':
    texts = getTexts(urls)
    logging.info('Length: %s', len(texts))
    for t in texts:
        print(t)
